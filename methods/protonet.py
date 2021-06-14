# This code is modified from https://github.com/jakesnell/prototypical-networks

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from model_resnet import *
from itertools import cycle

import wandb

class ProtoNetModel(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, jigsaw=False, lbda=0.0, rotation=False, tracking=False, use_bn=True, pretrain=False):
        super(ProtoNetModel, self).__init__(model_func,  n_way, n_support, use_bn, pretrain, tracking=tracking)
        self.loss_fn = nn.CrossEntropyLoss()

        self.jigsaw = jigsaw
        self.rotation = rotation
        self.lbda = lbda
        self.global_count = 0

        if self.jigsaw:
            self.fc6 = nn.Sequential()
            self.fc6.add_module('fc6_s1',nn.Linear(512, 512))#for resnet
            self.fc6.add_module('relu6_s1',nn.ReLU(inplace=False))
            self.fc6.add_module('drop6_s1',nn.Dropout(p=0.5))

            self.fc7 = nn.Sequential()
            self.fc7.add_module('fc7',nn.Linear(9*512,4096))#for resnet
            self.fc7.add_module('relu7',nn.ReLU(inplace=False))
            self.fc7.add_module('drop7',nn.Dropout(p=0.5))

            self.classifier = nn.Sequential()
            self.classifier.add_module('fc8',nn.Linear(4096, 35))
        if self.rotation:
            self.fc6 = nn.Sequential()
            self.fc6.add_module('fc6_s1',nn.Linear(512, 512))#for resnet
            self.fc6.add_module('relu6_s1',nn.ReLU(inplace=False))
            self.fc6.add_module('drop6_s1',nn.Dropout(p=0.5))

            self.fc7 = nn.Sequential()
            self.fc7.add_module('fc7',nn.Linear(512,128))#for resnet
            self.fc7.add_module('relu7',nn.ReLU(inplace=False))
            self.fc7.add_module('drop7',nn.Dropout(p=0.5))

            self.classifier_rotation = nn.Sequential()
            self.classifier_rotation.add_module('fc8',nn.Linear(128, 4))
    
    def forward(self,x, fc6=False, fc7=False, classifier=False, classifier_rotation=False):
        if fc6:
            return self.fc6(x)
        elif fc7:
            return self.fc7(x)
        elif classifier:
            return self.classifier(x)
        elif classifier_rotation:
            return self.classifier_rotation(x)

        return self.feature(x)
class ProtoNet(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, jigsaw=False, lbda=0.0, rotation=False, tracking=False, use_bn=True, pretrain=False):
        super(ProtoNet, self).__init__(model_func,  n_way, n_support, use_bn, pretrain, tracking=tracking)
        self.loss_fn = nn.CrossEntropyLoss()

        self.jigsaw = jigsaw
        self.rotation = rotation
        self.lbda = lbda
        self.global_count = 0
        self.model = ProtoNetModel(model_func,  n_way, n_support, jigsaw, lbda, rotation, tracking, use_bn, pretrain)
            
    def train_loop(self, epoch, train_loader, optimizer, base_loader_u=None, gpu=0):
        print_freq = 10
        avg_loss=0
        avg_loss_proto=0
        avg_loss_jigsaw=0
        avg_loss_rotation=0

        self.global_count = epoch * len(train_loader)
        
        # Uncomment the below and shift, add an else: when using unlabelled
        
#         if base_loader_u is not None:

#             for i,inputs in enumerate(zip(train_loader,cycle(base_loader_u))):
#                 self.global_count += 1
#                 x = inputs[0][0]
#                 self.n_query = x.size(1) - self.n_support
#                 if self.change_way:
#                     self.n_way  = x.size(0)
#                 optimizer.zero_grad()
#                 loss_proto, acc = self.set_forward_loss(x)
#                 if self.jigsaw:
#                     loss_jigsaw, acc_jigsaw = self.set_forward_loss_unlabel(inputs[1][2], inputs[1][3])# torch.Size([5, 21, 9, 3, 75, 75]), torch.Size([5, 21])
#                     loss = (1.0-self.lbda) * loss_proto + self.lbda * loss_jigsaw
#                     wandb.log({'train/loss_proto': float(loss_proto.data.item())}, step=self.global_count)
#                     wandb.log({'train/loss_jigsaw': float(loss_jigsaw.data.item())}, step=self.global_count)

#                 elif self.rotation:
#                     loss_rotation, acc_rotation = self.set_forward_loss_unlabel(inputs[1][2], inputs[1][3])# torch.Size([5, 21, 9, 3, 75, 75]), torch.Size([5, 21])
#                     loss = (1.0-self.lbda) * loss_proto + self.lbda * loss_rotation
#                     wandb.log({'train/loss_proto': float(loss_proto.data.item())}, step=self.global_count)
#                     wandb.log({'train/loss_rotation': float(loss_rotation.data.item())}, step=self.global_count)

#                 else:
#                     loss = loss_proto
#                 loss.backward()
#                 optimizer.step()
#                 avg_loss = avg_loss+loss.data
#                 wandb.log({'train/loss': float(loss.data.item())}, step=self.global_count)

#                 if self.jigsaw:
#                     avg_loss_proto += loss_proto.data
#                     avg_loss_jigsaw += loss_jigsaw.data
#                     wandb.log({'train/acc_proto': acc}, step=self.global_count)
#                     wandb.log({'train/acc_jigsaw': acc_jigsaw}, step=self.global_count)
#                 elif self.rotation:
#                     avg_loss_proto += loss_proto.data
#                     avg_loss_rotation += loss_rotation.data
#                     wandb.log({'train/acc_proto': acc}, step=self.global_count)
#                     wandb.log({'train/acc_rotation': acc_rotation}, step=self.global_count)
#                 if (i+1) % print_freq==0:
#                     if self.jigsaw:
#                         print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Loss Proto {:f} | Loss Jigsaw {:f}'.\
#                             format(epoch, i+1, len(train_loader), avg_loss/float(i+1), avg_loss_proto/float(i+1), avg_loss_jigsaw/float(i+1)))
#                     elif self.rotation:
#                         print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Loss Proto {:f} | Loss Rotation {:f}'.\
#                             format(epoch, i+1, len(train_loader), avg_loss/float(i+1), avg_loss_proto/float(i+1), avg_loss_rotation/float(i+1)))
#                     else:
#                         print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i+1, len(train_loader), avg_loss/float(i+1)))
#         else:
        for i, inputs in enumerate(train_loader):
            self.global_count += 1
            x = inputs[0]
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way  = x.size(0)
            optimizer.zero_grad()
            loss_proto, acc = self.set_forward_loss(x)
            if self.jigsaw:
                loss_jigsaw, acc_jigsaw = self.set_forward_loss_unlabel(inputs[2], inputs[3])# torch.Size([5, 21, 9, 3, 75, 75]), torch.Size([5, 21])
                loss = (1.0-self.lbda) * loss_proto + self.lbda * loss_jigsaw
                avg_loss_proto += loss_proto.data
                avg_loss_jigsaw += loss_jigsaw.data
                if not gpu:
                    wandb.log({'train/loss_jigsaw': float(loss_jigsaw.data.item())}, step=self.global_count)
                    wandb.log({'train/acc_jigsaw': acc_jigsaw}, step=self.global_count)
            elif self.rotation:
                loss_rotation, acc_rotation = self.set_forward_loss_unlabel(inputs[2], inputs[3])# torch.Size([5, 21, 9, 3, 75, 75]), torch.Size([5, 21])
                loss = (1.0-self.lbda) * loss_proto + self.lbda * loss_rotation
                avg_loss_proto += loss_proto.data
                avg_loss_rotation += loss_rotation.data
                if not gpu:
                    wandb.log({'train/loss_rotation': float(loss_rotation.data.item())}, step=self.global_count)
                    wandb.log({'train/acc_rotation': acc_rotation}, step=self.global_count)
            else:
                loss = loss_proto            

            loss.backward()
            optimizer.step()
            avg_loss = avg_loss+loss.item()

            if not gpu:
                wandb.log({'train/loss_proto': float(loss_proto.data.item())}, step=self.global_count)
                wandb.log({'train/acc_proto': acc}, step=self.global_count)
                wandb.log({'train/loss': float(loss.data.item())}, step=self.global_count)

            if (i+1) % print_freq==0 and not gpu:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                if self.jigsaw:
                    print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Loss Proto {:f} | Loss Jigsaw {:f}'.\
                        format(epoch, i+1, len(train_loader), avg_loss/float(i+1), avg_loss_proto/float(i+1), avg_loss_jigsaw/float(i+1)))
                elif self.rotation:
                    print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Loss Proto {:f} | Loss Rotation {:f}'.\
                        format(epoch, i+1, len(train_loader), avg_loss/float(i+1), avg_loss_proto/float(i+1), avg_loss_rotation/float(i+1)))
                else:
                    print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i+1, len(train_loader), avg_loss/float(i+1)))

                        
    def test_loop_with_loss(self, test_loader, record = None):
        correct =0
        count = 0
        acc_all = []
        acc_all_jigsaw = []
        acc_all_rotation = []

        avg_loss=0
        avg_loss_proto=0
        avg_loss_jigsaw=0
        avg_loss_rotation=0
        
        iter_num = len(test_loader)
        for i, inputs in enumerate(test_loader):
            x = inputs[0]
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way  = x.size(0)

            if self.jigsaw:
                correct_this, correct_this_jigsaw, count_this, count_this_jigsaw, loss_proto, loss_jigsaw = self.correct_with_loss(x, inputs[2], inputs[3])
                loss = (1.0-self.lbda) * loss_proto + self.lbda * loss_jigsaw
                avg_loss_proto += loss_proto.data
                avg_loss_jigsaw += loss_jigsaw.data
            elif self.rotation:
                correct_this, correct_this_rotation, count_this, count_this_rotation, loss_proto, loss_rotation = self.correct_with_loss(x, inputs[2], inputs[3])
                loss = (1.0-self.lbda) * loss_proto + self.lbda * loss_rotation
                avg_loss_proto += loss_proto.data
                avg_loss_rotation += loss_rotation.data
            else:
                correct_this, count_this, loss_proto = self.correct_with_loss(x)
                avg_loss_proto += loss_proto.data
                loss = loss_proto
                
            avg_loss = avg_loss+loss.data
            
            acc_all.append(correct_this/ count_this*100)
            if self.jigsaw:
                acc_all_jigsaw.append(correct_this_jigsaw/ count_this_jigsaw*100)
            elif self.rotation:
                acc_all_rotation.append(correct_this_rotation/ count_this_rotation*100)

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        
        avg_loss = avg_loss / iter_num
        avg_loss_proto = avg_loss_proto / iter_num
        avg_loss_jigsaw = avg_loss_jigsaw / iter_num
        avg_loss_rotation = avg_loss_rotation / iter_num
        
        print('%d Test Protonet Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
        if self.jigsaw:
            acc_all_jigsaw  = np.asarray(acc_all_jigsaw)
            acc_mean_jigsaw = np.mean(acc_all_jigsaw)
            acc_std_jigsaw  = np.std(acc_all_jigsaw)
            print('%d Test Jigsaw Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean_jigsaw, 1.96* acc_std_jigsaw/np.sqrt(iter_num)))
            return (acc_mean, acc_mean_jigsaw), (avg_loss, avg_loss_proto, avg_loss_jigsaw)
        elif self.rotation:
            acc_all_rotation  = np.asarray(acc_all_rotation)
            acc_mean_rotation = np.mean(acc_all_rotation)
            acc_std_rotation  = np.std(acc_all_rotation)
            print('%d Test Rotation Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean_rotation, 1.96* acc_std_rotation/np.sqrt(iter_num)))
            return (acc_mean, acc_mean_rotation), (avg_loss, avg_loss_proto, avg_loss_rotation)
        else:
            return (acc_mean), (avg_loss, avg_loss_proto)
                        
    def test_loop(self, test_loader, record = None):
        correct =0
        count = 0
        acc_all = []
        acc_all_jigsaw = []
        acc_all_rotation = []

        iter_num = len(test_loader)
        for i, inputs in enumerate(test_loader):
            x = inputs[0]
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way  = x.size(0)

            if self.jigsaw:
                correct_this, correct_this_jigsaw, count_this, count_this_jigsaw = self.correct(x, inputs[2], inputs[3])
            elif self.rotation:
                correct_this, correct_this_rotation, count_this, count_this_rotation = self.correct(x, inputs[2], inputs[3])
            else:
                correct_this, count_this = self.correct(x)
            acc_all.append(correct_this/ count_this*100)
            if self.jigsaw:
                acc_all_jigsaw.append(correct_this_jigsaw/ count_this_jigsaw*100)
            elif self.rotation:
                acc_all_rotation.append(correct_this_rotation/ count_this_rotation*100)

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Protonet Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
        if self.jigsaw:
            acc_all_jigsaw  = np.asarray(acc_all_jigsaw)
            acc_mean_jigsaw = np.mean(acc_all_jigsaw)
            acc_std_jigsaw  = np.std(acc_all_jigsaw)
            print('%d Test Jigsaw Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean_jigsaw, 1.96* acc_std_jigsaw/np.sqrt(iter_num)))
            return acc_mean, acc_mean_jigsaw
        elif self.rotation:
            acc_all_rotation  = np.asarray(acc_all_rotation)
            acc_mean_rotation = np.mean(acc_all_rotation)
            acc_std_rotation  = np.std(acc_all_rotation)
            print('%d Test Rotation Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean_rotation, 1.96* acc_std_rotation/np.sqrt(iter_num)))
            return acc_mean, acc_mean_rotation
        else:
            return acc_mean

    def correct(self, x, patches=None, patches_label=None):
        scores = self.set_forward(x)
        if self.jigsaw:
            x_, y_ = self.set_forward_unlabel(patches=patches,patches_label=patches_label)
        elif self.rotation:
            x_, y_ = self.set_forward_unlabel(patches=patches,patches_label=patches_label)
        y_query = np.repeat(range( self.n_way ), self.n_query )

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)

        if self.jigsaw:
            pred = torch.max(x_,1)
            top1_correct_jigsaw = torch.sum(pred[1] == y_)
            return float(top1_correct), float(top1_correct_jigsaw), len(y_query), len(y_)
        elif self.rotation:
            pred = torch.max(x_,1)
            top1_correct_rotation = torch.sum(pred[1] == y_)
            return float(top1_correct), float(top1_correct_rotation), len(y_query), len(y_)
        else:
            return float(top1_correct), len(y_query)
        
    def correct_with_loss(self, x, patches=None, patches_label=None):
        scores = self.set_forward(x)
        if self.jigsaw:
            x_, y_ = self.set_forward_unlabel(patches=patches,patches_label=patches_label)
            jigsaw_loss = self.loss_fn(x_,y_)
        elif self.rotation:
            x_, y_ = self.set_forward_unlabel(patches=patches,patches_label=patches_label)
            rotation_loss = self.loss_fn(x_,y_)
            
        y_query = np.repeat(range( self.n_way ), self.n_query )

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)

        y_query_ = Variable(torch.from_numpy(y_query).cuda())
        loss = self.loss_fn(scores, y_query_)
        
        if self.jigsaw:
            pred = torch.max(x_,1)
            top1_correct_jigsaw = torch.sum(pred[1] == y_)
            return float(top1_correct), float(top1_correct_jigsaw), len(y_query), len(y_), loss, jigsaw_loss
        elif self.rotation:
            pred = torch.max(x_,1)
            top1_correct_rotation = torch.sum(pred[1] == y_)
            return float(top1_correct), float(top1_correct_rotation), len(y_query), len(y_), loss, rotation_loss
        else:
            return float(top1_correct), len(y_query), loss

    def set_forward(self,x,is_feature = False):
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous()
        z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores

    def set_forward_unlabel(self, patches=None, patches_label=None):
        if len(patches.size()) == 6:
            Way,S,T,C,H,W = patches.size()#torch.Size([5, 15, 9, 3, 75, 75])
            B = Way*S
        elif len(patches.size()) == 5:
            B,T,C,H,W = patches.size()#torch.Size([5, 15, 9, 3, 75, 75])
        if self.jigsaw:
            patches = patches.view(B*T,C,H,W).cuda()#torch.Size([675, 3, 64, 64])
            if self.dual_cbam:
                patch_feat = self.model.forward(patches, jigsaw=True)#torch.Size([675, 512])
            else:
                patch_feat = self.model.forward(patches)

            x_ = patch_feat.view(B,T,-1)
            x_ = x_.transpose(0,1)#torch.Size([9, 75, 512])

            x_list = []
            for i in range(9):
                z = self.model.forward(x_[i], fc6=True)#torch.Size([75, 512])
                z = z.view([B,1,-1])#torch.Size([75, 1, 512])
                x_list.append(z)

            x_ = torch.cat(x_list,1)#torch.Size([75, 9, 512])
            x_ = self.model.forward(x_.view(B,-1), fc7=True)#torch.Size([75, 9*512])
            x_ = self.model.forward(x_, classifier=True)

            y_ = patches_label.view(-1).cuda()

            return x_, y_
        elif self.rotation:
            patches = patches.view(B*T,C,H,W).cuda()
            x_ = self.model.forward(patches)
            x_ = x_.squeeze()
            x_ = self.model.forward(x_, fc6=True)
            x_ = self.model.forward(x_, fc7=True)#64,128
            x_ = self.model.forward(x_, classifier_rotation=True)#64,4
            pred = torch.max(x_,1)
            y_ = patches_label.view(-1).cuda()
            return x_, y_


    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        scores = self.set_forward(x)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        acc = np.sum(topk_ind[:,0] == y_query.numpy())/len(y_query.numpy())
        y_query = Variable(y_query.cuda())

        return self.loss_fn(scores, y_query), acc

    def set_forward_loss_unlabel(self, patches=None, patches_label=None):
        if self.jigsaw:
            x_, y_ = self.set_forward_unlabel(patches=patches,patches_label=patches_label)
            pred = torch.max(x_,1)
            acc_jigsaw = torch.sum(pred[1] == y_).cpu().numpy()*1.0/len(y_)
        elif self.rotation:
            x_, y_ = self.set_forward_unlabel(patches=patches,patches_label=patches_label)
            pred = torch.max(x_,1)
            acc_rotation = torch.sum(pred[1] == y_).cpu().numpy()*1.0/len(y_)

        if self.jigsaw:
            return self.loss_fn(x_,y_), acc_jigsaw
        elif self.rotation:
            return self.loss_fn(x_,y_), acc_rotation


    def parse_feature(self,x,is_feature):
        x    = Variable(x.cuda())
        if is_feature:
            z_all = x
        else:
            x           = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:])
            z_all       = self.model.forward(x)

            z_all       = z_all.view( self.n_way, self.n_support + self.n_query, -1)
        z_support   = z_all[:, :self.n_support]
        z_query     = z_all[:, self.n_support:]

        return z_support, z_query



def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
