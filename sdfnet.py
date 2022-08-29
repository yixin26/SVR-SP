import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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

def conv(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0)
        )

def conv_relu(in_planes, out_planes, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                  stride=stride, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True)
        )

def conv_tanh(in_planes, out_planes, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                  stride=stride, padding=(kernel_size-1)//2),
        nn.Tanh()
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

def offset_predictor(in_dim,out_dim):
    conv_planes = [in_dim, 512, 256 ]
    conv1 = conv_relu(conv_planes[0], conv_planes[1], kernel_size=1)
    conv2 = conv_relu(conv_planes[1], conv_planes[2], kernel_size=1)
    convout = conv_tanh(conv_planes[2], out_dim, kernel_size=1)
    return nn.Sequential(conv1,conv2,convout)

def camera_projection(pc, trans_mat):
    # sample_pc B*N*3
    B,N = pc.size()[0:2]
    homo_pc = torch.cat([pc, torch.ones(B, N, 1).to(pc.get_device())], dim=-1)
    pc_xyz = torch.matmul(homo_pc, trans_mat)
    pc_xy = torch.div(pc_xyz[:,:,:2], pc_xyz[:,:,2].unsqueeze(2))
    return torch.clamp(pc_xy,0.0,136.0)

class SDFNet(nn.Module):
    def __init__(self, config):
        super(SDFNet, self).__init__()
        self.config = config
        self.imgsize = config.img_h

        vgg_pretrained = True
        self.vgg16_feature = vgg16_feature(pretrained=vgg_pretrained)
        self.vgg16_fc = vgg16_fc(imgsize=self.imgsize)

        self.pointfeat = point_feature()
        pointfeat_dim = 512

        self.ref_points_num = 6
        vgg_local_dim = [64,128,256,512,512,512]
        vgg_glob_dim = 1024
        self.offset_predict = offset_predictor(pointfeat_dim*(1+self.ref_points_num), self.ref_points_num*3)
        self.agg_5 = conv_relu(vgg_local_dim[5]*(self.ref_points_num+1),vgg_local_dim[5], kernel_size=1)
        self.agg_4 = conv_relu(vgg_local_dim[4]*(self.ref_points_num+1),vgg_local_dim[4], kernel_size=1)
        self.agg_3 = conv_relu(vgg_local_dim[3]*(self.ref_points_num+1),vgg_local_dim[3], kernel_size=1)
        self.agg_2 = conv_relu(vgg_local_dim[2]*(self.ref_points_num+1),vgg_local_dim[2], kernel_size=1)
        self.agg_1 = conv_relu(vgg_local_dim[1]*(self.ref_points_num+1),vgg_local_dim[1], kernel_size=1)
        self.agg_0 = conv_relu(vgg_local_dim[0]*(self.ref_points_num+1),vgg_local_dim[0], kernel_size=1)

        self.cls_glob = sdf_regressor(vgg_glob_dim+pointfeat_dim)
        self.cls_local = sdf_regressor(sum(vgg_local_dim)+pointfeat_dim)

        self.init_weights([self.vgg16_fc,self.pointfeat,self.cls_glob,self.cls_local,self.agg_5,self.agg_4,self.agg_3,self.agg_2,self.agg_1,self.agg_0,self.offset_predict])

    def init_weights(self, nets):
        for net in nets:
            for m in net:
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    nn.init.xavier_uniform_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()
                if isinstance(m, nn.Linear):
                    m.weight.data.normal_()

    def get_image_feature_maps(self, net, img):
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

        H,W = img.shape[2:4]

        fms = []
        img = img.contiguous()
        for i in range(len(net)):
            img = net[i](img)
            if i in select_index:
                fms += [F.interpolate(img, size=[H,W], mode='bilinear', align_corners=False)]

        return img , fms

    def get_local_image_features(self, fms, pixs, pixs_ref):

        local_feats = []
        local_feats_ref = []
        s=float(self.imgsize)//2

        if len(pixs)>=1:
            t = []
            for px in pixs:
                t += [((px-s)/s).unsqueeze(2)]
            pixs = t

        if len(pixs_ref)>=1:
            t = []
            for px in pixs_ref:
                t += [((px-s)/s).unsqueeze(2)]
            pixs_ref = t

        for x in fms:
            #point sampler;
            if len(pixs) >= 1:
                x1 = []
                for px in pixs:
                    x1 += [F.grid_sample(x, px, mode='bilinear', align_corners=True)]
                local_feats += [torch.cat(x1, dim=1)]

            if len(pixs_ref) >= 1:
                x_ref = []
                for px in pixs_ref:
                    x_ref += [F.grid_sample(x, px, mode='bilinear', align_corners=True)]
                local_feats_ref += [torch.cat(x_ref, dim=1)]

        return local_feats, local_feats_ref

    def get_affinities_uniform(self, pts):
        pts_aff = []
        radius = 0.1
        pts_aff += [torch.cat([pts[:, :, 0:2], pts[:, :, 2:3]+radius], dim=2)]
        pts_aff += [torch.cat([pts[:, :, 0:1]+radius,pts[:,:,1:3]], dim=2)]
        pts_aff += [torch.cat([pts[:, :, 0:1], pts[:, :, 1:2]+radius,pts[:, :, 2:3]], dim=2)]
        pts_aff += [torch.cat([pts[:, :, 0:2], pts[:, :, 2:3]-radius], dim=2)]
        pts_aff += [torch.cat([pts[:, :, 0:1]-radius,pts[:,:,1:3]], dim=2)]
        pts_aff += [torch.cat([pts[:, :, 0:1], pts[:, :, 1:2]-radius,pts[:, :, 2:3]], dim=2)]
        return pts_aff

    def get_affinities_non_uniform(self, pts):
        pts_aff = []
        pts_aff += [torch.cat([pts[:, :, 0:2], -pts[:, :, 2:3]], dim=2)]                # [x,y,-z]
        pts_aff += [torch.cat([-pts[:, :, 0:1],pts[:,:,1:3]], dim=2)]                   # [-x,y,z]
        pts_aff += [torch.cat([pts[:, :, 0:1], -pts[:, :, 1:2],pts[:, :, 2:3]], dim=2)] # [x,-y,z]
        pts_aff += [torch.cat([-pts[:, :, 0:2], pts[:, :, 2:3]], dim=2)]                # [-x,-y,z]
        pts_aff += [torch.cat([pts[:, :, 0:1],-pts[:,:,1:3]], dim=2)]                   # [x,-y,-z]
        pts_aff += [torch.cat([-pts[:, :, 0:1], pts[:, :, 1:2],-pts[:, :, 2:3]], dim=2)]# [-x,y,-z]
        return pts_aff

    def get_affinities(self, pts):
        return self.get_affinities_non_uniform(pts)

    def modify_affinities(self, pts_aff, offsets):
        pts_aff[0] += offsets[:, :, 0:3]
        pts_aff[1] += offsets[:, :, 3:6]
        pts_aff[2] += offsets[:, :, 6:9]
        pts_aff[3] += offsets[:, :, 9:12]
        pts_aff[4] += offsets[:, :, 12:15]
        pts_aff[5] += offsets[:, :, 15:18]
        return pts_aff

    def forward(self, img, pts, trans_mat):

        ptsfeat = self.pointfeat(pts.transpose(2, 1).unsqueeze(3))  # B * C * N * 1;  pts:B,N,3
        lastfeat, featuremaps = self.get_image_feature_maps(self.vgg16_feature,img)
        glob_feat = self.vgg16_fc(lastfeat)  # B * C * 1 * 1
        glob_feat = glob_feat.repeat(1, 1, pts.size(1), 1)  # B * C * N * 1

        # get sampled pixels according to sampled 3d points and the camera pose.
        pixs = camera_projection(pts, trans_mat)  # B * N * 2
        localfeats,_ = self.get_local_image_features(featuremaps, [pixs], [])

        #3d affinity sampling
        pts_aff = self.get_affinities(pts)

        ptsfeats_ref = []
        for i in range(self.ref_points_num):
            ptsfeats_ref += [self.pointfeat(pts_aff[i].transpose(2, 1).unsqueeze(3))]

        #adjust affinities
        offsets = self.offset_predict(torch.cat([ptsfeat]+ptsfeats_ref, dim=1)).squeeze(-1).transpose(2, 1)

        pts_aff = self.modify_affinities(pts_aff,offsets)

        #get 2d projections
        pixs_ref = []
        for i in range(self.ref_points_num):
            pixs_ref += [camera_projection(torch.clamp(pts_aff[i], -1.0, 1.0), trans_mat)]
        _, localfeats_ref = self.get_local_image_features(featuremaps, [], pixs_ref)

        # local feature aggregation
        sub_lf5 = self.agg_5(torch.cat([localfeats[5], localfeats_ref[5]], dim=1))
        sub_lf4 = self.agg_4(torch.cat([localfeats[4], localfeats_ref[4]], dim=1))
        sub_lf3 = self.agg_3(torch.cat([localfeats[3], localfeats_ref[3]], dim=1))
        sub_lf2 = self.agg_2(torch.cat([localfeats[2], localfeats_ref[2]], dim=1))
        sub_lf1 = self.agg_1(torch.cat([localfeats[1], localfeats_ref[1]], dim=1))
        sub_lf0 = self.agg_0(torch.cat([localfeats[0], localfeats_ref[0]], dim=1))
        local_feat = torch.cat([sub_lf5, sub_lf4, sub_lf3, sub_lf2, sub_lf1, sub_lf0], dim=1)

        sdf_glob = self.cls_glob(torch.cat([glob_feat, ptsfeat], dim=1))
        sdf_local = self.cls_local(torch.cat([local_feat, ptsfeat], dim=1))

        pred_sdf = sdf_glob + sdf_local
        return pred_sdf.squeeze(1).squeeze(-1)

    def get_aff(self, pts):
        pts_aff = self.get_affinities(pts)
        ptsfeat = self.pointfeat(pts.transpose(2, 1).unsqueeze(3))  # B * C * N * 1;  pts:B,N,3
        ptsfeats_ref = []
        for i in range(self.ref_points_num):
            ptsfeats_ref += [self.pointfeat(pts_aff[i].transpose(2, 1).unsqueeze(3))]
        offsets = self.offset_predict(torch.cat([ptsfeat]+ptsfeats_ref, dim=1)).squeeze(-1).transpose(2, 1)
        pts_aff = self.modify_affinities(pts_aff,offsets)
        return torch.cat(pts_aff, dim=2)
