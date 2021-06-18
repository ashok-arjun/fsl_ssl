import backbone
import utils

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from model_resnet import *
from resnet_pytorch import *

import wandb

from io_utils import data_prefetcher

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

class BaselineTrain(nn.Module):
    def __init__(self, model_func, num_class, loss_type = 'softmax', jigsaw=False, lbda=0.0, rotation=False, tracking=True, pretrain=False):
        super(BaselineTrain, self).__init__()
        self.jigsaw = jigsaw
        self.lbda = lbda
        self.rotation = rotation
        self.tracking = tracking
        print('tracking in baseline train:',tracking)
        self.pretrain = pretrain
        print("USE pre-trained model:",pretrain)

        if isinstance(model_func,str):
            if model_func == 'resnet18':
                self.feature = ResidualNet('ImageNet', 18, 1000, None, tracking=self.tracking)
                self.feature.final_feat_dim = 512
            elif model_func == 'resnet18_pytorch':
                self.feature = resnet18(pretrained=self.pretrain, tracking=self.tracking)
                self.feature.final_feat_dim = 512
            elif model_func == 'resnet50_pytorch':
                self.feature = resnet50(pretrained=self.pretrain, tracking=self.tracking)
                self.feature.final_feat_dim = 2048
        else:
            self.feature = model_func()

        if loss_type == 'softmax':
            self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
            self.classifier.bias.data.fill_(0)
        elif loss_type == 'dist': #Baseline ++
            self.classifier = backbone.distLinear(self.feature.final_feat_dim, num_class)
        self.loss_type = loss_type  #'softmax' #'dist'
        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()
        self.global_count = 0

        if self.jigsaw or self.rotation:
            self.fc6 = nn.Sequential()
            self.fc6.add_module('fc6_s1',nn.Linear(512, 512))#for resnet
            self.fc6.add_module('relu6_s1',nn.ReLU(inplace=True))
            self.fc6.add_module('drop6_s1',nn.Dropout(p=0.5))
            
        if self.jigsaw:

            self.fc7_jigsaw = nn.Sequential()
            self.fc7_jigsaw.add_module('fc7',nn.Linear(9*512,4096))#for resnet
            self.fc7_jigsaw.add_module('relu7',nn.ReLU(inplace=True))
            self.fc7_jigsaw.add_module('drop7',nn.Dropout(p=0.5))

            self.classifier_jigsaw = nn.Sequential()
            self.classifier_jigsaw.add_module('fc8',nn.Linear(4096, 35))

        if self.rotation:

            self.fc7_rotation = nn.Sequential()
            self.fc7_rotation.add_module('fc7',nn.Linear(512,128))#for resnet
            self.fc7_rotation.add_module('relu7',nn.ReLU(inplace=True))
            self.fc7_rotation.add_module('drop7',nn.Dropout(p=0.5))

            self.classifier_rotation = nn.Sequential()
            self.classifier_rotation.add_module('fc8',nn.Linear(128, 4))

    def forward(self,x):
        x    = Variable(x.cuda())
        out  = self.feature(x)
        scores  = self.classifier(out.view(x.size(0), -1))
        return scores

    def forward_loss(self, x=None, y=None, patches=None, patches_label=None, unlabel_only=False, label_only=False):
        # import ipdb; ipdb.set_trace()
        if not unlabel_only:
            scores = self.forward(x)
            y = Variable(y.cuda())
            pred = torch.argmax(scores, dim=1)

            if torch.cuda.is_available():
                acc = (pred == y).type(torch.cuda.FloatTensor).mean().item()
            else:
                acc = (pred == y).type(torch.FloatTensor).mean().item()

        if label_only:
            return self.loss_fn(scores, y), acc

        if self.jigsaw:
            B,T,C,H,W = patches.size()#torch.Size([16, 9, 3, 64, 64])
            patches = patches.view(B*T,C,H,W).cuda()#torch.Size([144, 3, 64, 64])
            patch_feat = self.feature(patches)#torch.Size([144, 512, 1, 1])

            x_ = patch_feat.view(B,T,-1)#torch.Size([16, 9, 512])
            x_ = x_.transpose(0,1)#torch.Size([9, 16, 512])

            x_list = []
            for i in range(9):
                z = self.fc6(x_[i])#torch.Size([16, 512])
                z = z.view([B,1,-1])#torch.Size([16, 1, 512])
                x_list.append(z)

            x_ = torch.cat(x_list,1)#torch.Size([16, 9, 512])
            x_ = self.fc7_jigsaw(x_.view(B,-1))#torch.Size([16, 9*512])
            x_ = self.classifier_jigsaw(x_)

            y_ = patches_label.view(-1).cuda()

            pred = torch.max(x_,1)
            acc_jigsaw = torch.sum(pred[1] == y_).cpu().numpy()*1.0/len(y_)
            if unlabel_only:
                return self.loss_fn(x_,y_), acc_jigsaw
            else:
                return self.loss_fn(scores, y), self.loss_fn(x_,y_), acc, acc_jigsaw
        elif self.rotation:
            B,R,C,H,W = patches.size()#torch.Size([16, 4, 3, 224, 224])
            patches = patches.view(B*R,C,H,W).cuda()
            x_ = self.feature(patches)#torch.Size([64, 512, 1, 1])
            x_ = x_.squeeze()
            x_ = self.fc6(x_)
            x_ = self.fc7_rotation(x_)#64,128
            x_ = self.classifier_rotation(x_)#64,4
            pred = torch.max(x_,1)
            y_ = patches_label.view(-1).cuda()
            acc_jigsaw = torch.sum(pred[1] == y_).cpu().numpy()*1.0/len(y_)
            if unlabel_only:
                return self.loss_fn(x_,y_), acc_jigsaw
            else:
                return self.loss_fn(scores, y), self.loss_fn(x_,y_), acc, acc_jigsaw
        else:
            return self.loss_fn(scores, y), acc
    
    def train_loop(self, epoch, train_loader, optimizer, scheduler=None, base_loader_u=None, pbar=None, enable_amp=None):
        print_freq = len(train_loader)
        avg_loss=0
        avg_loss_softmax=0
        avg_loss_jigsaw=0
        avg_loss_rotation=0
        avg_acc_softmax=0
        avg_acc_jigsaw=0
        avg_acc_rotation=0

#         if base_loader_u is not None:
#             for i,inputs in enumerate(zip(train_loader,base_loader_u)):
#                 self.global_count += 1
#                 x = inputs[0][0]
#                 y = inputs[0][1]
#                 optimizer.zero_grad()
#                 loss_softmax, acc = self.forward_loss(x, y, label_only=True)

#                 if self.jigsaw:
#                     loss_jigsaw, acc_jigsaw = self.forward_loss(patches=inputs[1][2], patches_label=inputs[1][3], unlabel_only=True)
#                     loss = (1.0-self.lbda) * loss_softmax + self.lbda * loss_jigsaw
#                     wandb.log({'train/loss_softmax': float(loss_softmax.data.item())}, step=self.global_count)
#                     wandb.log({'train/loss_jigsaw': float(loss_jigsaw.data.item())}, step=self.global_count)
#                 elif self.rotation:
#                     loss_rotation, acc_rotation = self.forward_loss(patches=inputs[1][2], patches_label=inputs[1][3], unlabel_only=True)
#                     loss = (1.0-self.lbda) * loss_softmax + self.lbda * loss_rotation
#                     wandb.log({'train/loss_softmax': float(loss_softmax.data.item())}, step=self.global_count)
#                     wandb.log({'train/loss_rotation': float(loss_rotation.data.item())}, step=self.global_count)
#                 else:
#                     loss, acc = self.forward_loss(x,y)
#                 wandb.log({'train/loss': float(loss.data.item())}, step=self.global_count)
#                 loss.backward()
#                 optimizer.step()

#                 if scheduler is not None:
#                     scheduler.step()
#                     wandb.log({'train/lr': optimizer.param_groups[0]['lr']}, step=self.global_count)

#                 avg_loss = avg_loss+loss.data#[0]
#                 avg_acc_softmax = avg_acc_softmax+acc

#                 wandb.log({'train/acc_cls': acc}, step=self.global_count)

#                 if self.jigsaw:
#                     avg_loss_softmax += loss_softmax.data
#                     avg_loss_jigsaw += loss_jigsaw
#                     avg_acc_jigsaw = avg_acc_jigsaw+acc_jigsaw
#                     wandb.log({'train/acc_jigsaw': acc_jigsaw}, step=self.global_count)
#                 elif self.rotation:
#                     avg_loss_softmax += loss_softmax.data
#                     avg_loss_rotation += loss_rotation
#                     avg_acc_rotation = avg_acc_rotation+acc_rotation
#                     wandb.log({'train/acc_rotation': acc_rotation}, step=self.global_count)


#                 if (i+1) % print_freq==0:
#                     if self.jigsaw:
#                         print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Loss Cls {:f} | Loss Jigsaw {:f} | Acc Cls {:f} | Acc Jigsaw {:f}'.\
#                             format(epoch+1, i+1, len(train_loader), avg_loss/float(i+1), avg_loss_softmax/float(i+1), \
#                                     avg_loss_jigsaw/float(i+1), avg_acc_softmax/float(i+1), avg_acc_jigsaw/float(i+1)))
#                     elif self.rotation:
#                         print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Loss Cls {:f} | Loss Rotation {:f} | Acc Cls {:f} | Acc Rotation {:f}'.\
#                             format(epoch+1, i+1, len(train_loader), avg_loss/float(i+1), avg_loss_softmax/float(i+1), \
#                                     avg_loss_rotation/float(i+1), avg_acc_softmax/float(i+1), avg_acc_rotation/float(i+1)))
#                     else:
#                         print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Acc Cls {:f}'.format(epoch+1, i+1, \
#                                         len(train_loader), avg_loss/float(i+1), avg_acc_softmax/float(i+1)  ))

#         else:
        i = 0 
        max_iter = len(train_loader)
        while i < max_iter:
            try:
                inputs = iter_loader.next()
            except:
                iter_loader = data_prefetcher(train_loader)
                inputs = iter_loader.next()
            self.global_count += 1
            x = inputs[0]
            y = inputs[1]
            optimizer.zero_grad()
            if self.jigsaw:
                loss_softmax, loss_jigsaw, acc, acc_jigsaw = self.forward_loss(x, y, inputs[2], inputs[3])
                loss = (1.0-self.lbda) * loss_softmax + self.lbda * loss_jigsaw
                wandb.log({'train/loss_softmax': float(loss_softmax.data.item())}, step=self.global_count)
                wandb.log({'train/loss_jigsaw': float(loss_jigsaw.data.item())}, step=self.global_count)

                avg_loss_softmax += loss_softmax.data
                avg_loss_jigsaw += loss_jigsaw
                avg_acc_jigsaw = avg_acc_jigsaw+acc_jigsaw
                wandb.log({'train/acc_jigsaw': acc_jigsaw}, step=self.global_count)
                wandb.log({'train/acc_softmax': acc}, step=self.global_count)

            elif self.rotation:
                loss_softmax, loss_rotation, acc, acc_rotation = self.forward_loss(x, y, inputs[2], inputs[3])
                loss = (1.0-self.lbda) * loss_softmax + self.lbda * loss_rotation
                wandb.log({'train/loss_softmax': float(loss_softmax.data.item())}, step=self.global_count)
                wandb.log({'train/loss_rotation': float(loss_rotation.data.item())}, step=self.global_count)


                avg_loss_softmax += loss_softmax.data
                avg_loss_rotation += loss_rotation
                avg_acc_rotation = avg_acc_rotation+acc_rotation
                wandb.log({'train/acc_rotation': acc_rotation}, step=self.global_count)
                wandb.log({'train/acc_softmax': acc}, step=self.global_count)
            else:
                loss, acc = self.forward_loss(x,y)
                wandb.log({'train/loss_softmax': float(loss.data.item())}, step=self.global_count)
                wandb.log({'train/acc_softmax': acc}, step=self.global_count)

            avg_loss = avg_loss+loss.data
            avg_acc_softmax = avg_acc_softmax+acc                
            wandb.log({'train/loss': float(loss.data.item())}, step=self.global_count)


            if enable_amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            optimizer.step()

            if scheduler is not None:
                scheduler.step()
                wandb.log({'train/lr': optimizer.param_groups[0]['lr']}, step=self.global_count)     
                
            pbar.update(1)
            
            if (i+1) % print_freq==0:
                if self.jigsaw:
                    pbar.write('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Loss Cls {:f} | Loss Jigsaw {:f} | Acc Cls {:f} | Acc Jigsaw {:f}'.\
                        format(epoch+1, i+1, len(train_loader), avg_loss/float(i+1), avg_loss_softmax/float(i+1), \
                                avg_loss_jigsaw/float(i+1), avg_acc_softmax/float(i+1), avg_acc_jigsaw/float(i+1)))
                elif self.rotation:
                    pbar.write('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Loss Cls {:f} | Loss Rotation {:f} | Acc Cls {:f} | Acc Rotation {:f}'.\
                        format(epoch+1, i+1, len(train_loader), avg_loss/float(i+1), avg_loss_softmax/float(i+1), \
                                avg_loss_rotation/float(i+1), avg_acc_softmax/float(i+1), avg_acc_rotation/float(i+1)))
                else:
                    pbar.write('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Acc Cls {:f}'.format(epoch+1, i+1, \
                                    len(train_loader), avg_loss/float(i+1), avg_acc_softmax/float(i+1)  ))
            i += 1
                         
    def test_loop(self, val_loader=None, pbar=None):
        if val_loader is not None:
            num_correct = 0
            num_total = 0
            num_correct_jigsaw = 0
            num_total_jigsaw = 0
            iter_num = len(val_loader)
            i = 0 
            while i < iter_num:
                i += 1 
                try:
                    inputs = iter_loader.next()
                except:
                    iter_loader = data_prefetcher(val_loader)
                    inputs = iter_loader.next()
                x = inputs[0]
                y = inputs[1]
                if self.jigsaw:
                    loss_proto, loss_jigsaw, acc, acc_jigsaw = self.forward_loss(x, y, inputs[2], inputs[3])
                    loss = (1.0-self.lbda) * loss_proto + self.lbda * loss_jigsaw
                    num_correct_jigsaw = int(acc_jigsaw*len(inputs[3]))
                    num_total_jigsaw += len(inputs[3].view(-1))
                elif self.rotation:
                    loss_proto, loss_rotation, acc, acc_rotation = self.forward_loss(x, y, inputs[2], inputs[3])
                    loss = (1.0-self.lbda) * loss_proto + self.lbda * loss_rotation
                    num_correct_jigsaw = int(acc_jigsaw*len(inputs[3]))
                    num_total_jigsaw += len(inputs[3].view(-1))
                else:
                    loss, acc = self.forward_loss(x,y)
                num_correct += int(acc*x.shape[0])
                num_total += len(y)
            
            if self.jigsaw or self.rotation:
                return num_correct*100.0/num_total, num_correct_jigsaw*100.0/num_total_jigsaw
            else:
                return num_correct*100.0/num_total


    def test_loop_with_loss(self, val_loader=None):
        num_correct = 0
        num_total = 0
        num_correct_jigsaw = 0
        num_total_jigsaw = 0
        num_correct_rotation= 0
        num_total_rotation = 0

        avg_loss=0
        avg_loss_softmax=0
        avg_loss_jigsaw=0
        avg_loss_rotation=0

        for i, inputs in enumerate(val_loader):
            x = inputs[0]
            y = inputs[1]
            if self.jigsaw:
                loss_softmax, loss_jigsaw, acc, acc_jigsaw = self.forward_loss(x, y, inputs[2], inputs[3])
                loss = (1.0-self.lbda) * loss_softmax + self.lbda * loss_jigsaw
                num_correct_jigsaw = int(acc_jigsaw*len(inputs[3]))
                num_total_jigsaw += len(inputs[3].view(-1))
                avg_loss_jigsaw += loss_jigsaw
            elif self.rotation:
                loss_softmax, loss_rotation, acc, acc_rotation = self.forward_loss(x, y, inputs[2], inputs[3])
                loss = (1.0-self.lbda) * loss_softmax + self.lbda * loss_rotation
                num_correct_rotation = int(acc_rotation*len(inputs[3]))
                num_total_rotation += len(inputs[3].view(-1))
                avg_loss_rotation += loss_rotation
            else:
                loss, acc = self.forward_loss(x,y)
                loss_softmax = loss

            num_correct += int(acc*x.shape[0])
            num_total += len(y)

            avg_loss += loss
            avg_loss_softmax += loss_softmax

        avg_loss /= len(val_loader)
        avg_loss_softmax /= len(val_loader)
        avg_loss_rotation /= len(val_loader)
        avg_loss_jigsaw /= len(val_loader)

        if self.jigsaw:
            return (num_correct*100.0/num_total, num_correct_jigsaw*100.0/num_total_jigsaw), (avg_loss, avg_loss_softmax, avg_loss_jigsaw)
        elif self.rotation:
            return (num_correct*100.0/num_total, num_correct_rotation*100.0/num_total_rotation), (avg_loss, avg_loss_softmax, avg_loss_rotation)
        else:
            return (num_correct*100.0/num_total), (avg_loss, avg_loss_softmax)
