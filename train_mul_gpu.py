# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
from torch.utils.tensorboard import SummaryWriter   
# import neptune
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import matplotlib
import random
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from PIL import Image
import copy
import time
import os
from model import ft_net_YOSOViM
from random_erasing import RandomErasing
from autoaugment import ImageNetPolicy, CIFAR10Policy
import yaml
import math
from shutil import copyfile
from utils import update_average, get_model_list, load_network, save_network, make_weights_for_balanced_classes
import numpy as np
from image_folder import SatData, DroneData, ImageFolder_selectID, ImageFolder_expandID
from timm.scheduler.cosine_lr import CosineLRScheduler
import torch.nn.functional as F
version =  torch.__version__
#fp16
try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------
def str2bool(v):
    """
    Converts string to bool type; enables command line
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='debug', type=str, help='output model name')
parser.add_argument('--pool',default='avg', type=str, help='pool avg')
parser.add_argument('--data_dir',default='/home/wangtyu/datasets/University-Release/train',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--batchsize', default=16, type=int, help='batchsize')
parser.add_argument('--stride', default=1, type=int, help='stride')
parser.add_argument('--pad', default=10, type=int, help='padding')
parser.add_argument('--h', default=256, type=int, help='height')
parser.add_argument('--w', default=256, type=int, help='width')
parser.add_argument('--views', default=2, type=int, help='the number of views')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--use_NAS', action='store_true', help='use NAS' )
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--moving_avg', default=1.0, type=float, help='moving average')
parser.add_argument('--droprate', default=0.75, type=float, help='drop rate')
parser.add_argument('--DA', action='store_true', help='use Color Data Augmentation' )
parser.add_argument('--resume', action='store_true', help='use resume trainning' )
parser.add_argument('--share', action='store_true', help='share weight between different view' )
parser.add_argument('--extra_Google', action='store_true', help='using extra noise Google' )
parser.add_argument('--LPN', action='store_true', help='use LPN' )
parser.add_argument('--decouple', action='store_true', help='use decouple' )
parser.add_argument('--block', default=4, type=int, help='the num of block' )
parser.add_argument('--scale', default=1/32, type=float, metavar='S', help='scale the loss')
parser.add_argument('--lambd', default=3.9e-3, type=float, metavar='L', help='weight on off-diagonal terms')
parser.add_argument('--g', default=0.9, type=float, metavar='L', help='weight on loss and deloss')
parser.add_argument('--t', default=4.0, type=float, metavar='L', help='temperature of conv matrix')
parser.add_argument('--experiment_name',default='debug',type=str, help='log dir name')
parser.add_argument('--adam', action='store_true', help='using adam optimization' )
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--balance', action='store_true', help='using balance sampler' )
parser.add_argument('--select_id', action='store_true', help='select id' )
parser.add_argument('--multi_image', action='store_true', help='only inputs3 + inputs3_s training' )
parser.add_argument('--expand_id', action='store_true', help='expand id' )
parser.add_argument('--dro_lead', action='store_true', help='drone leading sampling' )
parser.add_argument('--sat_lead', action='store_true', help='satellite leading sampling' )
parser.add_argument('--normal', action='store_true', help='normal training' )
parser.add_argument('--only_decouple', action='store_true', help='do not use balance losss' )
parser.add_argument('--e1', default=1, type=int, help='the exponent of on diag' )
parser.add_argument('--e2', default=1, type=int, help='the exponent of off diag' )
parser.add_argument('--swin', action='store_true', help='using swin as backbone' )
parser.add_argument('--yosovim', action='store_true', help='using MobileMamba as backbone' )
parser.add_argument('--fp16', action='store_true', help='use float16 instead of float32, which will save about 50% memory' )
parser.add_argument('--norm', type=str2bool, default=True)
opt = parser.parse_args()
print('------------------------------yosovim:',opt.yosovim)
def seed_torch(seed=opt.seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	# torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
if opt.seed > 0:
    print('random seed---------------------:', opt.seed)
    seed_torch(opt.seed)

if opt.resume:
    model, opt, start_epoch = load_network(opt.name, opt)
else:
    start_epoch = 0

# debug
# opt.LPN=True
# opt.decouple = True




fp16 = opt.fp16
data_dir = opt.data_dir
name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids)>1:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids 
    cudnn.enabled = True
    cudnn.benchmark = True
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str,gpu_ids))
    cudnn.benchmark = True
print('---------------Pool Strategy------------:', opt.pool)
######################################################################
# Load Data
# ---------
#

transform_train_list = [
        #transforms.RandomResizedCrop(size=(opt.h, opt.w), scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.Pad( opt.pad, padding_mode='edge'),
        transforms.RandomCrop((opt.h, opt.w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

transform_satellite_list = [
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.Pad( opt.pad, padding_mode='edge'),
        transforms.RandomAffine(90),
        transforms.RandomCrop((opt.h, opt.w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

transform_val_list = [
        transforms.Resize(size=(opt.h, opt.w),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

if opt.erasing_p>0:
    transform_train_list = transform_train_list +  [RandomErasing(probability = opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list
    transform_satellite_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_satellite_list

if opt.DA:
    transform_train_list = [ImageNetPolicy()] + transform_train_list

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose( transform_train_list ),
    'val': transforms.Compose(transform_val_list),
    'satellite': transforms.Compose(transform_satellite_list)
    }


train_all = ''
if opt.train_all:
     train_all = '_all'

image_datasets = {}
if opt.expand_id:
    print('--------------------expand id-----------------------')
    image_datasets['satellite'] = ImageFolder_expandID(os.path.join(data_dir, 'satellite'), transform=data_transforms['satellite'])
else:
    image_datasets['satellite'] = SatData(data_dir, data_transforms['satellite'], data_transforms['train'])

if opt.select_id:
    print('--------------------select id-----------------------')
    image_datasets['drone'] = ImageFolder_selectID(os.path.join(data_dir, 'drone'), transform=data_transforms['train'])
else:
    image_datasets['drone'] = DroneData(data_dir, data_transforms['train'], data_transforms['satellite'])

def _init_fn(worker_id):
    np.random.seed(int(opt.seed)+worker_id)
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                            shuffle=True, num_workers=8, pin_memory=False, worker_init_fn=_init_fn) # 8 workers may work faster
            for x in ['satellite', 'drone']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['satellite', 'drone']}
class_names = image_datasets['satellite'].classes
print(dataset_sizes)
use_gpu = torch.cuda.is_available()

######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

# work channel loss
#def kl_loss(mu, std):
   # kl = 0.5 * torch.sum(mu**2 + std**2 - torch.log(std**2 + 1e-8) - 1, dim=-1)
    #return kl.mean()
class TripletLoss(nn.Module):
    def __init__(self, margin=None, norm=True):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.norm = norm
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets):
        with torch.autocast(enabled=False, device_type="cuda"):
            return self._forward(inputs.float(), targets)

    def _forward(self, inputs, targets):
        n = inputs.size(0)
        if self.norm:
            inputs = nn.functional.normalize(inputs, p=2.0, dim=1)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist = torch.addmm(dist, inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i] == 1].max().view(1, 1))
            dist_an.append(dist[i][mask[i] == 0].min().view(1, 1))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss


class AdaLoss(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, gamma=0.10):
        super(AdaLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, labels):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            labels: ground truth labels with shape (num_classes)
        """
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs]
        num = len(inputs)
        log_pts = []
        pts = []
        for i in range(num):
            #print(f"inputs[{i}].shape = {inputs[i].shape}")
            log_pt = self.logsoftmax(inputs[i])
            log_pts.append(log_pt)
            pts.append(log_pt.data.exp())
        targets = torch.zeros(log_pts[0].size(), requires_grad=False).scatter_(1, labels.unsqueeze(1).data.cpu(), 1)
        targets = targets.to(inputs[0].device)

        for i in range(num):
            if i == 0:
                loss = -(targets * log_pts[i]).mean(0).sum()
            if i > 0:
                if self.gamma == 0:
                    loss += -(targets * log_pts[i]).mean(0).sum()
                else:
                    indexs = torch.repeat_interleave(labels.unsqueeze(1), inputs[0].shape[1], dim=1)
                    predict = torch.gather(pts[i - 1], 1, indexs)
                    ada_weight = (1 - predict).pow(self.gamma)
                    ada_weight = ada_weight / ada_weight.mean(dim=0, keepdim=True)
                    loss += -(ada_weight * targets * log_pts[i]).mean(0).sum()
        return loss

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def decouple_loss(y1, y2, scale_loss, lambd):
    batch_size = y1.size(0)
    c = y1.T @ y2
    c.div_(batch_size)
    on_diag = torch.diagonal(c)
    p_on = (1 - on_diag) / 2 * 2
    on_diag = torch.pow(p_on, opt.e1) * torch.pow(torch.diagonal(c).add_(-1), 2)
    on_diag = on_diag.sum().mul(scale_loss)

    off_diag = off_diagonal(c)
    p_off = torch.abs(off_diag) * 2
    off_diag = torch.pow(p_off, opt.e2) * torch.pow(off_diagonal(c), 2)
    off_diag = off_diag.sum().mul(scale_loss)
    loss = on_diag + off_diag * lambd
    return loss, on_diag, off_diag * lambd


def one_LPN_output(outputs, labels, criterion, block):
    # part = {}
    sm = nn.Softmax(dim=1)
    num_part = block
    score = 0
    loss = 0
    for i in range(num_part):
        part = outputs[i]
        score += sm(part)
        loss += criterion(part, labels)

    _, preds = torch.max(score.data, 1)

    return preds, loss 

def train_model(model, model_test, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    criterion_ce = nn.CrossEntropyLoss()
    criterion_ada = AdaLoss(gamma=0.1, num_classes=opt.nclasses).cuda()
    criterion_tri = TripletLoss().cuda()
    w_lpn = 1.0  # LPN分支分类损失的权重
    w_ada = 1.0  # AdaLoss (RPM+Global) 的权重
    w_tri = 0.5  # Triplet Loss 的权重
    w_aug = 0.5  # 增强视图(balance模式)的总损失权重
    warm_up = 0.1
    warm_iteration = round(dataset_sizes['satellite'] / opt.batchsize) * opt.warm_epoch

    for epoch in range(num_epochs - start_epoch):
        epoch = epoch + start_epoch
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train']:
            model.train(True)

            running_loss = 0.0
            running_corrects = 0.0
            running_corrects3 = 0.0

            for data, data3 in zip(dataloaders['satellite'], dataloaders['drone']):
                # a. 数据准备
                inputs, inputs_d, labels = data
                inputs3, inputs3_s, labels3 = data3

                now_batch_size, c, h, w = inputs.shape
                if now_batch_size < opt.batchsize:
                    continue

                inputs, labels = inputs.cuda(), labels.cuda()
                inputs3, labels3 = inputs3.cuda(), labels3.cuda()
                if opt.balance:
                    inputs_d, inputs3_s = inputs_d.cuda(), inputs3_s.cuda()

                optimizer.zero_grad()

                all_sat = model(inputs)
                all_dr = model(inputs3)

                def unpack(outputs, block):
                    lpn = outputs[:block]
                    rpm = outputs[block:block * 2]
                    glob = outputs[block * 2:block * 2 + 2]
                    return lpn, rpm, glob

                lpn_sat, rpm_sat, glob_sat = unpack(all_sat, opt.block)
                lpn_dr, rpm_dr, glob_dr = unpack(all_dr, opt.block)

                loss_lpn_sat = sum(criterion_ce(logit, labels) for _, logit in lpn_sat)

                logits_ada_sat = [logit for _, logit in rpm_sat] + [logit for _, logit in glob_sat]
                loss_ada_sat = criterion_ada(logits_ada_sat, labels)

                feat_tri_sat = torch.cat([feat for feat, _ in glob_sat], dim=1)
                loss_tri_sat = criterion_tri(feat_tri_sat, labels)

                total_loss_sat = w_lpn * loss_lpn_sat + w_ada * loss_ada_sat + w_tri * loss_tri_sat

                # =================== DRONE DOMAIN LOSS ===================
                # LPN Loss
                loss_lpn_dr = sum(criterion_ce(logit, labels3) for _, logit in lpn_dr)

                # AdaLoss
                logits_ada_dr = [logit for _, logit in rpm_dr] + [logit for _, logit in glob_dr]
                loss_ada_dr = criterion_ada(logits_ada_dr, labels3)

                # Triplet Loss
                feat_tri_dr = torch.cat([feat for feat, _ in glob_dr], dim=1)
                loss_tri_dr = criterion_tri(feat_tri_dr, labels3)

                # 组合无人机域的总损失
                total_loss_dr = w_lpn * loss_lpn_dr + w_ada * loss_ada_dr + w_tri * loss_tri_dr

                # 初始化最终损失
                loss = total_loss_sat + total_loss_dr

                # e. (可选) 计算增强视图的损失
                if opt.balance:
                    # 增强视图前向传播
                    all_sat_d = model(inputs_d)
                    all_dr_s = model(inputs3_s)

                    # 解包
                    lpn_sat_d, rpm_sat_d, glob_sat_d = unpack(all_sat_d, opt.block)
                    lpn_dr_s, rpm_dr_s, glob_dr_s = unpack(all_dr_s, opt.block)

                    # 计算增强视图的损失 (逻辑同上)
                    loss_lpn_sat_d = sum(criterion_ce(logit, labels) for _, logit in lpn_sat_d)
                    logits_ada_sat_d = [logit for _, logit in rpm_sat_d] + [logit for _, logit in glob_sat_d]
                    loss_ada_sat_d = criterion_ada(logits_ada_sat_d, labels)
                    feat_tri_sat_d = torch.cat([feat for feat, _ in glob_sat_d], dim=1)
                    loss_tri_sat_d = criterion_tri(feat_tri_sat_d, labels)
                    total_loss_sat_aug = w_lpn * loss_lpn_sat_d + w_ada * loss_ada_sat_d + w_tri * loss_tri_sat_d

                    loss_lpn_dr_s = sum(criterion_ce(logit, labels3) for _, logit in lpn_dr_s)
                    logits_ada_dr_s = [logit for _, logit in rpm_dr_s] + [logit for _, logit in glob_dr_s]
                    loss_ada_dr_s = criterion_ada(logits_ada_dr_s, labels3)
                    feat_tri_dr_s = torch.cat([feat for feat, _ in glob_dr_s], dim=1)
                    loss_tri_dr_s = criterion_tri(feat_tri_dr_s, labels3)
                    total_loss_dr_aug = w_lpn * loss_lpn_dr_s + w_ada * loss_ada_dr_s + w_tri * loss_tri_dr_s

                    # 将增强损失加权后计入总损失
                    loss += w_aug * (total_loss_sat_aug + total_loss_dr_aug)

                if epoch < opt.warm_epoch:
                    warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                    loss *= warm_up

                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()

                _, preds = torch.max(torch.mean(torch.stack([glob_sat[0][1], glob_sat[1][1]]), dim=0), 1)
                _, preds3 = torch.max(torch.mean(torch.stack([glob_dr[0][1], glob_dr[1][1]]), dim=0), 1)

                # statistics
                running_loss += loss.item() * now_batch_size
                if opt.decouple:
                    ins_loss += insloss.item() * now_batch_size
                    dec_loss += deloss.item() * now_batch_size
                    on_loss += on.item() * now_batch_size
                    off_loss += off.item() *now_batch_size

                running_corrects += float(torch.sum(preds == labels.data))                
                running_corrects3 += float(torch.sum(preds3 == labels3.data))

            epoch_loss = running_loss / (dataset_sizes['satellite']*2)
            epoch_acc = running_corrects / dataset_sizes['satellite']
            epoch_acc3 = running_corrects3 / dataset_sizes['satellite']

            if opt.decouple:
                epoch_ins_loss = ins_loss / dataset_sizes['satellite']
                epoch_dec_loss = dec_loss / dataset_sizes['satellite']
                epoch_on_loss = on_loss / dataset_sizes['satellite']
                epoch_off_loss = off_loss / dataset_sizes['satellite']
            
            if opt.decouple:
                print('{} Loss: {:.4f} Satellite_Acc: {:.4f} Drone_Acc: {:.4f}, On_Loss: {:.4f}, Off_Loss: {:.4f},'.format(phase, epoch_loss, epoch_acc, epoch_acc3, epoch_on_loss, epoch_off_loss))
            else:   
                print('{} Loss: {:.4f} Satellite_Acc: {:.4f} Drone_Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_acc3))
 
        
            writer.add_scalar('Train loss', epoch_loss, epoch+1)
            writer.add_scalar('Learning rate', optimizer.param_groups[1]['lr'], epoch+1)
            writer.add_scalar('Satellite Acc', epoch_acc, epoch+1)           
            writer.add_scalar('Drone Acc', epoch_acc3, epoch+1)
            if opt.decouple:
                writer.add_scalar('instance loss', epoch_ins_loss, epoch+1)
                writer.add_scalar('decouple loss', epoch_dec_loss, epoch+1)
                writer.add_scalar('on loss', epoch_on_loss, epoch+1)
                writer.add_scalar('off loss', epoch_off_loss, epoch+1)



            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0-epoch_acc)            
            
            # saving last model:
            if phase == 'train':
                scheduler.step()
            if epoch+1 == num_epochs and len(gpu_ids)>1:
                save_network(model.module, opt.name, epoch)
            elif epoch+1 > 200 and (epoch+1) % 100 == 0:
                save_network(model, opt.name, epoch)
            #draw_curve(epoch)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()
        # if epoch_loss < best_loss:
        #     best_loss = epoch_loss
        #     best_epoch = epoch
        #     last_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))
    # model.load_state_dict(last_model_wts)
    # if len(gpu_ids)>1:
    #     save_network(model.module, opt.name, 'last')
    #     print('best_epoch:', best_epoch)
    # else:
    #     save_network(model, opt.name, 'last')
    #     print('best_epoch:', best_epoch)

    return model


######################################################################
# Draw Curve
#---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig( os.path.join('./model',name,'train.jpg'))


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#
#if opt.LPN:
#    model = ft_net_LPN(len(class_names), droprate=opt.droprate, stride=opt.stride, pool=opt.pool, block=opt.block, decouple=opt.decouple)
if opt.swin:
    model = ft_net_swin(len(class_names), droprate=opt.droprate, decouple=opt.decouple)
#if opt.mmamba:
#    print('---------------using MobileMamba as backbone------------------')
#   model = ft_net_MMamba(len(class_names), droprate=opt.droprate, stride=opt.stride, pool=opt.pool, block=opt.block, decouple=opt.decouple)
if opt.yosovim:
    print('---------------using YOSOViM as backbone------------------')
    model = ft_net_YOSOViM(len(class_names), droprate=opt.droprate, stride=opt.stride, pool=opt.pool, block=opt.block, decouple=opt.decouple)
else:
    model = ft_net(len(class_names), droprate=opt.droprate, stride=opt.stride, pool=opt.pool, decouple=opt.decouple)

opt.nclasses = len(class_names)
print('nclass--------------------:', opt.nclasses)
print(model)
# For resume:
if start_epoch>=40:
    opt.lr = opt.lr*0.1
if not opt.LPN:
    model = model.cuda()
    ignored_params = list(map(id, model.classifier.parameters() ))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
                {'params': base_params, 'lr': 0.1*opt.lr},
                {'params': model.classifier.parameters(), 'lr': opt.lr}
            ], weight_decay=5e-4, momentum=0.9, nesterov=True)
else:
    # ignored_params = list(map(id, model.model.fc.parameters() ))
    if len(gpu_ids)>1:
        model = torch.nn.DataParallel(model).cuda()
        ignored_params = list()
        for i in range(opt.block):
            cls_name = 'classifier'+str(i)
            c = getattr(model.module, cls_name)
            ignored_params += list(map(id, c.parameters() ))
        
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

        optim_params = [{'params': base_params, 'lr': 0.1*opt.lr}]
        for i in range(opt.block):
            cls_name = 'classifier'+str(i)
            c = getattr(model.module, cls_name)
            optim_params.append({'params': c.parameters(), 'lr': opt.lr})

    else:
        model = model.cuda()
        print('---------------------use one gpu-----------------------')
        ignored_params =list()
        # ignored_params += list(map(id, model.rdim.parameters() ))
        for i in range(opt.block):
            cls_name = 'classifier'+str(i)
            c = getattr(model, cls_name)
            ignored_params += list(map(id, c.parameters() ))

        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

        optim_params = [{'params': base_params, 'lr': 0.1*opt.lr}]
        # optim_params.append({'params': model.rdim.parameters(), 'lr': opt.lr})
        for i in range(opt.block):
            cls_name = 'classifier'+str(i)
            c = getattr(model, cls_name)
            optim_params.append({'params': c.parameters(), 'lr': opt.lr})

    optimizer_ft = optim.SGD(optim_params, weight_decay=5e-4, momentum=0.9, nesterov=True)
    if opt.adam:
        optimizer_ft = optim.Adam(optim_params, opt.lr, weight_decay=5e-4)

# Decay LR by a factor of 0.1 every 40 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=80, gamma=0.1)
#exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[60,120,160], gamma=0.1)
exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=150, eta_min=0.001)
######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU. 
#
# neptune.init('wtyu/decouple')
# neptune.create_experiment('LPN+norm(batch*512*4)')

log_dir = './log/'+ opt.experiment_name
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)
writer = SummaryWriter(log_dir)
dir_name = os.path.join('./model',name)
if not opt.resume:
    if not os.path.isdir(dir_name):
        
        os.mkdir(dir_name)
#record every run
    copyfile('./run_mul_gpu_view.sh', dir_name+'/run_mul_gpu_view.sh')
    copyfile('./train_mul_gpu.py', dir_name+'/train_mul_gpu.py')
    copyfile('./model.py', dir_name+'/model.py')
# save opts
    with open('%s/opts.yaml'%dir_name,'w') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)

if fp16:
    model, optimizer_ft = amp.initialize(model, optimizer_ft, opt_level = "O1")

# if len(gpu_ids)>1:
#     model = torch.nn.DataParallel(model, device_ids=gpu_ids).cuda()
# else:
#     model = model.cuda()

criterion = nn.CrossEntropyLoss()
if opt.moving_avg<1.0:
    model_test = copy.deepcopy(model)
    num_epochs = 140
else:
    model_test = None
    num_epochs = 300

model = train_model(model, model_test, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=num_epochs)
# neptune.stop()
writer.close()