import torch
import torch.nn as nn
from sdfnet import get_model
import numpy as np
import os

def get_agent(config):
    return Agent(config)

def camera_projection(sample_pc, trans_mat_right):
    # sample_pc B*N*3
    B,N = sample_pc.size()[0:2]
    homo_pc = torch.cat([sample_pc, torch.ones(B, N, 1).to(sample_pc.get_device())], dim=-1)
    pc_xyz = torch.matmul(homo_pc, trans_mat_right)
    pc_xy = torch.div(pc_xyz[:,:,:2], pc_xyz[:,:,2].unsqueeze(2))
    return torch.clamp(pc_xy,0.0,136.0)

class Agent(object):
    def __init__(self, config):
        self.config = config
        self.model_dir = config.model_dir

        # build network
        self.net = self.build_net(config)

    def build_net(self, config):
        net = get_model(config)
        if config.parallel:
            net = nn.DataParallel(net)
        if config.use_gpu:
            net = net.cuda()
        return net

    def load_ckpt(self, name=None):
        """load checkpoint from saved checkpoint"""
        name = name if name == 'latest' else "ckpt_epoch{}".format(name)
        load_path = os.path.join(self.model_dir, "{}.pth".format(name))
        if not os.path.exists(load_path):
            raise ValueError("Checkpoint {} not exists.".format(load_path))

        checkpoint = torch.load(load_path)
        print("Checkpoint loaded from {}".format(load_path))
        if isinstance(self.net, nn.DataParallel):
            self.net.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.net.load_state_dict(checkpoint['model_state_dict'])

    def infer_func(self, data):
        self.net.eval()

        sample_pt = data['sdf_pt'].contiguous()
        img = data['img'].transpose(2, 3).transpose(1, 2).contiguous()
        trans_mat = data['trans_mat'].contiguous()
        if self.config.use_gpu:
            sample_pt = sample_pt.cuda()
            img = img.cuda()
            trans_mat = trans_mat.cuda()

        pixs = camera_projection(sample_pt, trans_mat)  # B * N * 2

        sample_pt_ref = sample_pt.clone()
        sample_pt_ref[:, :, 2] = -sample_pt_ref[:, :, 2] #xy plane reflection
        pixs_xy = camera_projection(sample_pt_ref, trans_mat)
        sample_pt_ref[:, :, 2] = -sample_pt_ref[:, :, 2]
        sample_pt_ref[:, :, 0] = -sample_pt_ref[:, :, 0]#yz plane reflection
        pixs_yz = camera_projection(sample_pt_ref, trans_mat)
        sample_pt_ref[:, :, 0] = -sample_pt_ref[:, :, 0]#yz plane reflection
        sample_pt_ref[:, :, 1] = -sample_pt_ref[:, :, 1]#xz plane reflection
        pixs_xz = camera_projection(sample_pt_ref, trans_mat)

        output_sdf = self.net(img, sample_pt, pixs, pixs_xy,pixs_yz,pixs_xz)
        return output_sdf