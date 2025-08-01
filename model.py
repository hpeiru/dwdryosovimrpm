import argparse
import math
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import timm
#from MobileMamba.model.mobilemamba.mobilemamba import MobileMamba_B1
from yosovim import YOSOViM

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

def fix_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

class LinearBlock(nn.Module):
    def __init__(self, input_dim, num_bottleneck=512):
        super(LinearBlock, self).__init__()
        self.Linear = nn.Linear(input_dim, num_bottleneck)
        init.kaiming_normal_(self.Linear.weight.data, a=0, mode='fan_out')
        init.constant_(self.Linear.bias.data, 0.0) 

    def forward(self, x):
        x = self.Linear(x)
        return x

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, mid_dim=256):
        super(ClassBlock, self).__init__()
        add_block = []
        self.Linear = nn.Linear(input_dim, num_bottleneck)
        self.bnorm = nn.BatchNorm1d(num_bottleneck)

        init.kaiming_normal_(self.Linear.weight.data, a=0, mode='fan_out')
        init.constant_(self.Linear.bias.data, 0.0) 
        init.normal_(self.bnorm.weight.data, 1.0, 0.02)
        init.constant_(self.bnorm.bias.data, 0.0)


        classifier = []
        if droprate>0:
            classifier += [nn.Dropout(p=droprate)]
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.classifier = classifier
    def forward(self, x):
        feat = self.Linear(x)
        feat_bn = self.bnorm(feat)
        logits = self.classifier(feat_bn)
        return feat_bn, logits  # 返回特征和分类分数

class LSMChannelBranch(nn.Module):
    def __init__(self, in_channels=384, feat_h=8, feat_w=8, block=4, class_num=701, droprate=0.5):
        super().__init__()
        self.block = block
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))  # GMP

        # 生成环形空间先验的 mask，初始为环状结构，可学习
        self.masks = nn.Parameter(self._init_ring_masks(block, feat_h, feat_w))  # shape: [4, 1, H, W]

        for i in range(block):
            setattr(self, f'classifier_rpm_{i}', ClassBlock(in_channels // block, class_num, droprate))

    def _init_ring_masks(self, block, H, W):
        center_h, center_w = H // 2, W // 2
        scale_factors = torch.linspace(1 / block, 1.0, block)
        masks = []
        prev_mask = torch.zeros(H, W)
        for i in range(block):
            rect_h_half = int(center_h * scale_factors[i])
            rect_w_half = int(center_w * scale_factors[i])
            y1, y2 = center_h - rect_h_half, center_h + rect_h_half
            x1, x2 = center_w - rect_w_half, center_w + rect_w_half

            current_mask = torch.zeros(H, W)
            current_mask[y1:y2, x1:x2] = 1.0
            ring_mask = current_mask - prev_mask
            masks.append(ring_mask)
            prev_mask = current_mask
        stacked = torch.stack(masks).unsqueeze(1)  # [block, 1, H, W]
        return stacked

    def forward(self, x):
        B, C, H, W = x.size()
        x_chunks = torch.chunk(x, self.block, dim=1)  # channel split
        out = []
        for i in range(self.block):
            mask = torch.sigmoid(self.masks[i])  # 可学习但保持在(0,1)
            x_i = x_chunks[i] * mask  # 每个特征组 × 对应编号 mask
            pooled = self.gmp(x_i).view(B, -1)
            classifier_i = getattr(self, f'classifier_rpm_{i}')
            feat, logits = classifier_i(pooled)  # 现在返回两个值
            out.append((feat, logits))  # 存储为元组
        return out

class GlobalPoolingBranch(nn.Module):
    def __init__(self, in_channels=384, class_num=701, droprate=0.5):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier_gap = ClassBlock(in_channels, class_num, droprate)
        self.classifier_gmp = ClassBlock(in_channels, class_num, droprate)

    def forward(self, x):
        out = []
        gap_feat = self.gap(x).view(x.size(0), -1)
        gmp_feat = self.gmp(x).view(x.size(0), -1)
        # 修改1: 调用分类器并获取特征和分类分数
        gap_feature, gap_logits = self.classifier_gap(gap_feat)
        gmp_feature, gmp_logits = self.classifier_gmp(gmp_feat)
        # 修改2: 存储为元组 (特征, 分类分数)
        out.append((gap_feature, gap_logits))
        out.append((gmp_feature, gmp_logits))
        return out

class ft_net_YOSOViM(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg', block=4, decouple=False):
        super(ft_net_YOSOViM, self).__init__()
        from yosovim import YOSOViM
        self.model = YOSOViM(dims=[96,192,384], layers=[3,7,3], state_dims=[32,16,8], mlp_ratio=3.0, kernel_size=3, act_type="gelu")
        self.pool = pool
        self.block = block
        self.decouple = decouple
        self.bn = nn.BatchNorm1d(512, affine=False)
        self.avg = nn.AdaptiveMaxPool2d((1, 1))

        pretrained_weights = torch.load('./checkpoint_M4_1.2G.pth', map_location="cpu")
        self.model.load_state_dict(pretrained_weights, strict=False)

        for i in range(self.block):
            clas = 'classifier'+str(i)
            setattr(self, clas, ClassBlock(384, class_num, droprate=droprate))

        self.rpm_branch = LSMChannelBranch(in_channels=384, feat_h=8, feat_w=8, block=block, class_num=class_num, droprate=droprate)
        self.global_branch = GlobalPoolingBranch(in_channels=384, class_num=class_num, droprate=droprate)

    def forward(self, x):
        x = self.model.first_conv(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)

        part_feat = self.get_part_pool(x)
        part_feat = part_feat.view(part_feat.size(0), part_feat.size(1), -1)
        out_lpn = []
        for i in range(self.block):
            part_feat_i = part_feat[:, :, i]
            classifier_i = getattr(self, f'classifier{i}')
            feat, logits = classifier_i(part_feat_i)  # 获取特征和分数
            out_lpn.append((feat, logits))
            # if not self.training:
            #     print(f"LPN branch {i} feature shape: {feat.shape}")
        out_rpm = self.rpm_branch(x)
        # if not self.training:
        #     for i, (feat, logits) in enumerate(out_rpm):
        #         print(f"RPM branch {i} feature shape: {feat.shape}")
        out_global = self.global_branch(x)
        # if not self.training:
        #     for i, (feat, logits) in enumerate(out_global):
        #         print(f"Global branch {i} feature shape: {feat.shape}")

        if self.training:
            return out_lpn + out_rpm + out_global
        else:
            # 提取所有特征向量用于测试
            feats = []
            # 遍历所有分支的输出
            for branch_output in out_lpn + out_rpm + out_global:
                feat, _ = branch_output  # 只取特征部分
                feats.append(feat)
                # print(f"Branch feature shape: {feat.shape}")
            feats = torch.cat(feats, dim=1)
            #print(f"Concatenated feature shape: {feats.shape}")
            return F.normalize(feats, dim=1)

    def get_part_pool(self, x, no_overlap=True):
        result = []
        H, W = x.size(2), x.size(3)
        c_h, c_w = int(H / 2), int(W / 2)
        per_h, per_w = H / (2 * self.block), W / (2 * self.block)
        if per_h < 1 and per_w < 1:
            new_H, new_W = H + (self.block - c_h) * 2, W + (self.block - c_w) * 2
            x = nn.functional.interpolate(x, size=[new_H, new_W], mode='bilinear', align_corners=True)
            H, W = x.size(2), x.size(3)
            c_h, c_w = int(H / 2), int(W / 2)
            per_h, per_w = H / (2 * self.block), W / (2 * self.block)
        per_h, per_w = math.floor(per_h), math.floor(per_w)
        for i in range(self.block):
            i = i + 1
            if i < self.block:
                x_curr = x[:, :, (c_h - i * per_h):(c_h + i * per_h), (c_w - i * per_w):(c_w + i * per_w)]
                x_pre = None
                if no_overlap and i > 1:
                    x_pre = x[:, :, (c_h - (i - 1) * per_h):(c_h + (i - 1) * per_h),
                            (c_w - (i - 1) * per_w):(c_w + (i - 1) * per_w)]
                    x_pad = F.pad(x_pre, (per_h, per_h, per_w, per_w), "constant", 0)
                    x_curr = x_curr - x_pad
                avgpool = self.avg_pool(x_curr, x_pre)
                result.append(avgpool)

            else:
                if no_overlap and i > 1:
                    x_pre = x[:, :, (c_h - (i - 1) * per_h):(c_h + (i - 1) * per_h),
                            (c_w - (i - 1) * per_w):(c_w + (i - 1) * per_w)]
                    pad_h = c_h - (i - 1) * per_h
                    pad_w = c_w - (i - 1) * per_w
                    # x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    if x_pre.size(2) + 2 * pad_h == H:
                        x_pad = F.pad(x_pre, (pad_h, pad_h, pad_w, pad_w), "constant", 0)
                    else:
                        ep = H - (x_pre.size(2) + 2 * pad_h)
                        x_pad = F.pad(x_pre, (pad_h + ep, pad_h, pad_w + ep, pad_w), "constant", 0)
                    x = x - x_pad
                avgpool = self.avg_pool(x, x_pre)
                result.append(avgpool)
        return torch.stack(result, dim=2)

    def avg_pool(self, x_curr, x_pre=None):
        h, w = x_curr.size(2), x_curr.size(3)
        if x_pre == None:
            h_pre = w_pre = 0.0
        else:
            h_pre, w_pre = x_pre.size(2), x_pre.size(3)
        pix_num = h * w - h_pre * w_pre
        avg = x_curr.flatten(start_dim=2).sum(dim=2).div_(pix_num)
        return avg

    def part_classifier(self, x):

        out_p = []
        for i in range(self.block):
            o_tmp = x[:, :, i].view(x.size(0), -1)
            name = 'classifier' + str(i)
            c = getattr(self, name)
            out_p.append(c(o_tmp))

        if not self.training:
            return torch.stack(out_p, dim=2)
        else:
            return out_p


'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it. 
    # net = two_view_net(701, droprate=0.5, pool='avg', stride=1, VGG16=True, LPN=True, block=8, decouple=True)
    # net = ft_net_swin_base(701, droprate=0.5, decouple=False)
    # net = three_view_net(701, droprate=0.5, stride=1, share_weight=True, VGG16=False, LPN=True, block=4, decouple=False)
    # net = ft_net_LPN(701,0.75,1,block=4)
    # net.eval()

    # net = ft_net_VGG16_LPN_R(701)
    # net = ft_net_cvusa_LPN(701, stride=1)
    # net = ft_net_swin(701, droprate=0.75, decouple=True)
    net = ft_net_YOSOViM(701)
    #net.eval()

    print(net)

    input = Variable(torch.FloatTensor(2, 3, 256, 256))
    output1= net(input)
    # output1,output2 = net(input,input)
    # output1,output2,output3 = net(input,input,input)
    # output1 = net(input,decouple=False)
    # print('net output size:')
    print(output1[0][1].shape)
    # print(output.shape)
    # for i in range(len(output1)):
    #     print(output1[i].shape)
    # x = torch.randn(2,512,8,8)
    # x_shape = x.shape
    # pool = AzimuthPool2d(x_shape, 8)
    # out = pool(x)
    # print(out.shape)
