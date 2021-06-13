import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob
import random
import datetime
import dill

import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file
from tensorboardX import SummaryWriter
import json
from model_resnet import *
from utils import RunningAverage

import wandb

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler as DS


def main(gpu=None, params=None):
    
    if params.parallel:
        rank = params.nr * params.gpus + gpu
        dist.init_process_group(backend='nccl', init_method='env://', world_size=params.world_size, rank=rank)
        torch.cuda.set_device(gpu)
        torch.cuda.manual_seed_all(params.seed)
        
        print("One process runs on gpu: ", gpu)
    else:
        params.world_size = None
        rank = None
        
    isAircraft = (params.dataset == 'aircrafts')    
        
    base_file = os.path.join('filelists', params.dataset, params.base+'.json')
    val_file   = os.path.join('filelists', params.dataset, 'val.json')
     
    image_size = params.image_size

    if params.method in ['baseline', 'baseline++'] :
        base_datamgr    = SimpleDataManager(image_size, batch_size = params.bs, jigsaw=params.jigsaw, rotation=params.rotation, isAircraft=isAircraft, parallel=params.parallel, rank=rank, world_size=params.world_size, sampler_seed=params.seed)
        base_loader     = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
        val_datamgr     = SimpleDataManager(image_size, batch_size = params.bs, jigsaw=params.jigsaw, rotation=params.rotation, isAircraft=isAircraft, parallel=params.parallel, rank=rank, world_size=params.world_size, sampler_seed=params.seed)
        val_loader      = val_datamgr.get_data_loader( val_file, aug = False)

        if params.dataset == 'CUB':
            params.num_classes = 200
        elif params.dataset == 'cars':
            params.num_classes = 196
        elif params.dataset == 'aircrafts':
            params.num_classes = 100
        elif params.dataset == 'dogs':
            params.num_classes = 120
        elif params.dataset == 'flowers':
            params.num_classes = 102
        elif params.dataset == 'miniImagenet':
            params.num_classes = 100
        elif params.dataset == 'tieredImagenet':
            params.num_classes = 608

        if params.method == 'baseline':
            model           = BaselineTrain( model_dict[params.model], params.num_classes, \
                                            jigsaw=params.jigsaw, lbda=params.lbda, rotation=params.rotation, tracking=params.tracking, gpu=gpu)
        elif params.method == 'baseline++':
            model           = BaselineTrain( model_dict[params.model], params.num_classes, \
                                    loss_type = 'dist', jigsaw=params.jigsaw, lbda=params.lbda, rotation=params.rotation, tracking=params.tracking, gpu=gpu)

    elif params.method in ['protonet','matchingnet','relationnet', 'relationnet_softmax', 'maml', 'maml_approx']:
        n_query = max(1, int(params.n_query * params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
 
        train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot, \
                                        jigsaw=params.jigsaw, lbda=params.lbda, rotation=params.rotation) 
        base_datamgr            = SetDataManager(image_size, n_query = n_query,  **train_few_shot_params, isAircraft=isAircraft, parallel=params.parallel, rank=rank, world_size=params.world_size, sampler_seed=params.seed)
        base_loader             = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
         
        test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot, \
                                        jigsaw=params.jigsaw, lbda=params.lbda, rotation=params.rotation) 
        val_datamgr             = SetDataManager(image_size, n_query = n_query, **test_few_shot_params, isAircraft=isAircraft, parallel=params.parallel, rank=rank, world_size=params.world_size, sampler_seed=params.seed)
        val_loader              = val_datamgr.get_data_loader( val_file, aug = False) 

        if params.method == 'protonet':
            model           = ProtoNet( model_dict[params.model], **train_few_shot_params, use_bn=(not params.no_bn), pretrain=params.pretrain, tracking=params.tracking)
        elif params.method == 'matchingnet':
            model           = MatchingNet( model_dict[params.model], **train_few_shot_params )
        elif params.method in ['relationnet', 'relationnet_softmax']:
            feature_model = lambda: model_dict[params.model]( flatten = False )
            loss_type = 'mse' if params.method == 'relationnet' else 'softmax'

            model           = RelationNet( feature_model, loss_type = loss_type , **train_few_shot_params )
        elif params.method in ['maml' , 'maml_approx']:
            backbone.ConvBlock.maml = True
            backbone.SimpleBlock.maml = True
            backbone.BottleneckBlock.maml = True
            backbone.ResNet.maml = True

            BasicBlock.maml = True
            Bottleneck.maml = True
            ResNet.maml = True

            model           = MAML(  model_dict[params.model], approx = (params.method == 'maml_approx') , **train_few_shot_params )

    else:
       raise ValueError('Unknown method')


    params.checkpoint_dir = 'ckpts/%s/%s_%s_%s' %(params.dataset, params.date, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    if not params.method  in ['baseline', 'baseline++']: 
        params.checkpoint_dir += '_%dway_%dshot_%dquery' %( params.train_n_way, params.n_shot, params.n_query)
    
    if params.dataset_unlabel is not None:
        params.checkpoint_dir += params.dataset_unlabel
        params.checkpoint_dir += str(params.bs)

    ## Track bn stats
    if params.tracking:
        params.checkpoint_dir += '_tracking'

    ## Add jigsaw
    if params.jigsaw:
        params.checkpoint_dir += '_jigsaw_lbda%.2f'%(params.lbda)
        params.checkpoint_dir += params.optimization

    ## Add rotation
    if params.rotation:
        params.checkpoint_dir += '_rotation_lbda%.2f'%(params.lbda)
        params.checkpoint_dir += params.optimization

    params.checkpoint_dir += '_lr%.4f'%(params.lr)
    if params.finetune:
        params.checkpoint_dir += '_finetune'

    if not params.parallel or params.parallel and gpu == 0: 
        print('Checkpoint path:',params.checkpoint_dir)
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    if params.method == 'maml' or params.method == 'maml_approx' :
        stop_epoch = params.stop_epoch * model.n_task #maml use multiple tasks in one update 

    if params.resume:
        resume_file = get_resume_file(params.checkpoint_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            model.model.load_state_dict(tmp['state'])
            del tmp
    elif params.warmup: #We also support warmup from pretrained baseline feature, but we never used in our paper
        baseline_checkpoint_dir = 'ckpts/%s/%s_%s' %(params.dataset, params.model, 'baseline')
        if params.train_aug:
            baseline_checkpoint_dir += '_aug'
        warmup_resume_file = get_resume_file(baseline_checkpoint_dir)
        tmp = torch.load(warmup_resume_file)
        if tmp is not None: 
            state = tmp['state']
            state_keys = list(state.keys())
            for i, key in enumerate(state_keys):
                if "feature." in key:
                    newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
                    state[newkey] = state.pop(key)
                else:
                    state.pop(key)
            model.model.feature.load_state_dict(state)
        else:
            raise ValueError('No warm_up file')
    
    if params.loadfile != '':
        print('Loading model from: ' + params.loadfile)
        checkpoint = torch.load(params.loadfile)
        ## remove last layer for baseline
        pretrained_dict = {k: v for k, v in checkpoint['state'].items() if 'classifier' not in k and 'loss_fn' not in k}
        print('Load model from:',params.loadfile)
        model.model.load_state_dict(pretrained_dict, strict=False)

    model.model = model.model.cuda(gpu)
    
    if params.parallel:
        if params.gpus > 1:
            model.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.model)
        model.model = nn.parallel.DistributedDataParallel(model.model, device_ids=[gpu], find_unused_parameters=True)
    
        
    json.dump(vars(params), open(params.checkpoint_dir+'/configs.json','w'))
    
    
    # Init WANDB
    
    if params.resume_wandb_id:
        print('Resuming from wandb ID: ', params.resume_wandb_id)
        wandb.init(config=vars(params), project="fsl_ssl", id=params.resume_wandb_id, resume=True)
    else:
        print('Fresh wandb run')
        wandb.init(config=vars(params), project="fsl_ssl")
  
    if params.optimization == 'Adam':
        optimizer = torch.optim.Adam(model.model.parameters(), lr=params.lr)
    elif params.optimization == 'SGD':
        optimizer = torch.optim.SGD(model.model.parameters(), lr=params.lr)
    elif params.optimization == 'Nesterov':
        optimizer = torch.optim.SGD(model.model.parameters(), lr=params.lr, nesterov=True, momentum=0.9, weight_decay=params.wd)
    else:
       raise ValueError('Unknown optimization, please define by yourself')

    eval_interval = params.eval_interval
    max_acc = 0       
    
    if not params.parallel or params.parallel and gpu ==0: 
        print("Evaluation will be done every %d epochs\n\n" % (eval_interval))
    accum_epoch_time = RunningAverage()

    for epoch in range(start_epoch,stop_epoch):
        if params.parallel:
            base_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)
       
        start_time = time.time()
        model.model.train()
        model.train_loop(epoch, base_loader, optimizer) # CHECKED 
        end_time = time.time()
        accum_epoch_time.update(end_time - start_time)
        eta = str(datetime.timedelta(seconds = int(accum_epoch_time() * (stop_epoch - epoch))))
        
        if not params.parallel or params.parallel and gpu ==0: 
            print('Epoch %d complete; eta: %s' % (epoch, eta))
            wandb.log({"Epoch time": accum_epoch_time(), "Epoch": epoch}, step=model.global_count)

            if epoch % eval_interval == True or epoch == stop_epoch - 1: 
                print("Evaluating...")
                model.model.eval()

                if not os.path.isdir(params.checkpoint_dir):
                    os.makedirs(params.checkpoint_dir)

                if params.jigsaw:
                    acc, acc_jigsaw = model.test_loop( val_loader)
                    wandb.log({"val/acc_jigsaw": acc_jigsaw}, step=model.global_count)
    #                 wandb.log({"val/loss_jigsaw": avg_loss_jigsaw}, step=model.global_count)
                elif params.rotation:
                    acc, acc_rotation = model.test_loop( val_loader)
                    wandb.log({"val/acc_rotation": acc_rotation}, step=model.global_count)
    #                 wandb.log({"val/loss_rotation": avg_loss_rotation}, step=model.global_count)
                else:    
                    acc = model.test_loop( val_loader)

    #             wandb.log({"val/loss_avg": avg_loss}, step=model.global_count)
    #             wandb.log({"val/loss_proto": avg_loss_proto}, step=model.global_count)
                wandb.log({"val/acc": acc}, step=model.global_count)

                if acc > max_acc : #for baseline and baseline++, we don't use validation here so we let acc = -1
                    print("best model! save...")
                    max_acc = acc
                    outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
                    torch.save({'epoch':epoch, 'state':model.state_dict(), 'optimizer': optimizer.state_dict()}, outfile)
                    wandb.save(outfile)

            if ((epoch+1) % params.save_freq==0) or (epoch==stop_epoch-1):
    #             outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
    #             torch.save({'epoch':epoch, 'state':model.state_dict(), 'optimizer': optimizer.state_dict()}, outfile)

                outfile_2 = os.path.join(params.checkpoint_dir, 'last_model.tar')
                torch.save({'epoch':epoch, 'state':model.state_dict(), 'optimizer': optimizer.state_dict()}, outfile_2)
                wandb.save(outfile_2)

    

if __name__=='__main__':
       
    params = parse_args('train')

    """Set seed"""
    seed = params.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
        
    if not params.committed:
        print("Commit the code and then execute with the --committed arg")
        exit()

    os.environ["CUDA_VISIBLE_DEVICES"] = params.devices
    
    print("params.parallel = ", params.parallel)
    
    if params.parallel:
        
        print("params.gpus = ", params.gpus)
        print("params.devices = ", params.devices)

        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29534'
        params.world_size = params.gpus * params.nodes
        mp.spawn(main, nprocs=params.gpus, args=(params,))
    else:
        main(gpu=0, params=params)