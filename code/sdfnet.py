import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch.utils.model_zoo as model_zoo
import torchvision.models as models

def get_model(config):
    return SDFNet(config)

def vgg16_feature(pretrained=True):
    vgg_16 = models.vgg16(pretrained=pretrained)
    return vgg_16.features

def vgg16_fc(cls_num=1024, imgsize=137,pretrained=False):
    if imgsize == 224 and pretrained:
        vgg_16 = models.vgg16(pretrained=pretrained)
        return vgg_16.classifier
    elif imgsize == 137:
        k =4
        # Use conv2d instead of fully_connected layers.
        classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=k, padding=0),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, kernel_size=1, padding=0),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Conv2d(4096, cls_num, kernel_size=1, padding=0)
        )
        return classifier

def conv_relu(in_planes, out_planes, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                  stride=stride, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True)
        )

def conv(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0)
        )

def point_feature():
    conv_planes = [64, 256, 512]
    conv1 = conv_relu(3, conv_planes[0], kernel_size=1)
    conv2 = conv_relu(conv_planes[0], conv_planes[1], kernel_size=1)
    convout = conv(conv_planes[1], conv_planes[2])
    return nn.Sequential(
        conv1,conv2,convout
    )

def sdf_regressor(fs):
    conv_planes = [fs, 512, 256 ]
    conv1 = conv_relu(conv_planes[0], conv_planes[1], kernel_size=1)
    conv2 = conv_relu(conv_planes[1], conv_planes[2], kernel_size=1)
    convout = conv(conv_planes[2], 1)
    return nn.Sequential(conv1,conv2,convout)

class SDFNet(nn.Module):
    def __init__(self, config):
        super(SDFNet, self).__init__()
        self.config = config
        self.imgsize = config.img_h

        vgg_pretrained = True
        self.vgg16_feature = vgg16_feature(pretrained=vgg_pretrained)
        self.vgg16_fc = vgg16_fc(imgsize=self.imgsize)
        self.pointfeat = point_feature()

        ref_points_num = 3
        vgg_fm = [64,128,256,512,512,512]
        self.conv_gamma_6 = conv_relu(vgg_fm[5]*ref_points_num, vgg_fm[5],kernel_size=1, stride=1)
        self.conv_beta_6 = conv_relu(vgg_fm[5]*ref_points_num, vgg_fm[5],kernel_size=1, stride=1)
        self.conv_gamma_5 = conv_relu(vgg_fm[4]*ref_points_num, vgg_fm[4],kernel_size=1, stride=1)
        self.conv_beta_5 = conv_relu(vgg_fm[4]*ref_points_num, vgg_fm[4],kernel_size=1, stride=1)
        self.conv_gamma_4 = conv_relu(vgg_fm[3]*ref_points_num, vgg_fm[3],kernel_size=1, stride=1)
        self.conv_beta_4 = conv_relu(vgg_fm[3]*ref_points_num, vgg_fm[3],kernel_size=1, stride=1)
        self.conv_gamma_3 = conv_relu(vgg_fm[2]*ref_points_num, vgg_fm[2],kernel_size=1, stride=1)
        self.conv_beta_3 = conv_relu(vgg_fm[2]*ref_points_num, vgg_fm[2],kernel_size=1, stride=1)
        self.conv_gamma_2 = conv_relu(vgg_fm[1]*ref_points_num, vgg_fm[1],kernel_size=1, stride=1)
        self.conv_beta_2 = conv_relu(vgg_fm[1]*ref_points_num, vgg_fm[1],kernel_size=1, stride=1)
        self.conv_gamma_1 = conv_relu(vgg_fm[0]*ref_points_num, vgg_fm[0],kernel_size=1, stride=1)
        self.conv_beta_1 = conv_relu(vgg_fm[0]*ref_points_num, vgg_fm[0],kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

        self.cls_glob = sdf_regressor(1024+512)
        self.cls_local = sdf_regressor(1984+512)

    def get_local_image_feature(self, net, img, pixs, pixs_xy,pixs_yz,pixs_xz):
        vgg_cfg_D =     [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        select_layer =  [0 , 1 ,  0  , 0  , 1  , 0  , 0  , 0  , 1  , 0  , 0  , 0  , 1  , 0  , 0  , 0  , 1  , 1  ] #six layers
        exact_layer = 0
        for i in range(len(vgg_cfg_D)):
            if vgg_cfg_D[i] == 'M':
                exact_layer += 1
            else:
                exact_layer += 2
            select_layer[i] *= exact_layer-2
        select_index = torch.where(torch.tensor(select_layer) > 0)
        select_index = torch.tensor(select_layer)[select_index[0]]

        local_feats = []
        local_feats_ref = []
        H,W = img.shape[2:4]
        s=float(self.imgsize)//2
        pixs = (pixs-s)/s #normalize index to [-1,1]
        pixs = pixs.unsqueeze(2)
        pixs_xy = (pixs_xy-s)/s
        pixs_xy = pixs_xy.unsqueeze(2)
        pixs_yz = (pixs_yz-s)/s
        pixs_yz = pixs_yz.unsqueeze(2)
        pixs_xz = (pixs_xz-s)/s
        pixs_xz = pixs_xz.unsqueeze(2)

        img = img.contiguous()
        for i in range(len(net)):
            img = net[i](img)
            if i in select_index:
                x = F.interpolate(img, size=[H,W], mode='bilinear', align_corners=False)
                #point sampler;
                x1 = F.grid_sample(x,pixs, mode='bilinear', align_corners=True)
                x2 = F.grid_sample(x,pixs_xy, mode='bilinear', align_corners=True)
                x3 = F.grid_sample(x,pixs_yz, mode='bilinear', align_corners=True)
                x4 = F.grid_sample(x,pixs_xz, mode='bilinear', align_corners=True)
                local_feats += [x1]
                local_feats_ref += [torch.cat([x2,x3,x4], dim=1)]

        return img , local_feats, local_feats_ref

    def modify_local_feat(self,x, gamma, beta):
        x = (gamma * x) + beta
        return self.relu(x)

    def forward(self, img, pts, pixs, pixs_xy,pixs_yz,pixs_xz):
        ps = pts.size()[1] #point num : N
        pts = pts.transpose(2,1) #B,N,3 -> B,3,N

        lastfeat, localfeats, localfeats_ref = self.get_local_image_feature(self.vgg16_feature,img,pixs, pixs_xy,pixs_yz,pixs_xz)

        sub_lf6 = self.modify_local_feat(localfeats[5], self.conv_gamma_6(localfeats_ref[5]), self.conv_beta_6(localfeats_ref[5]))
        sub_lf5 = self.modify_local_feat(localfeats[4], self.conv_gamma_5(localfeats_ref[4]), self.conv_beta_5(localfeats_ref[4]))
        sub_lf4 = self.modify_local_feat(localfeats[3], self.conv_gamma_4(localfeats_ref[3]), self.conv_beta_4(localfeats_ref[3]))
        sub_lf3 = self.modify_local_feat(localfeats[2], self.conv_gamma_3(localfeats_ref[2]), self.conv_beta_3(localfeats_ref[2]))
        sub_lf2 = self.modify_local_feat(localfeats[1], self.conv_gamma_2(localfeats_ref[1]), self.conv_beta_2(localfeats_ref[1]))
        sub_lf1 = self.modify_local_feat(localfeats[0], self.conv_gamma_1(localfeats_ref[0]), self.conv_beta_1(localfeats_ref[0]))
        local_feat = torch.cat([sub_lf6,sub_lf5, sub_lf4, sub_lf3, sub_lf2, sub_lf1], dim=1)

        glob_feat = self.vgg16_fc(lastfeat) # B * C * 1 * 1
        glob_feat = glob_feat.repeat(1, 1, ps, 1) # B * C * N * 1
        ptsfeat = self.pointfeat(pts.unsqueeze(3)) # B * C * N * 1

        sdf_glob = self.cls_glob(torch.cat([glob_feat, ptsfeat], dim=1))
        sdf_local = self.cls_local(torch.cat([local_feat, ptsfeat], dim=1))

        pred_sdf = sdf_glob + sdf_local
        return pred_sdf.squeeze(1).squeeze(-1)
