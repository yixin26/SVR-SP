import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from collections import OrderedDict
from tqdm import tqdm
from config import get_config
from agent import get_agent
import numpy as np
import random
from joblib import Parallel, delayed
import pymesh
import torch
import struct

RESOLUTION = 33
TOTAL_POINTS = RESOLUTION * RESOLUTION * RESOLUTION
SPLIT_SIZE = int(np.ceil(TOTAL_POINTS / 50000.0 ))
NUM_SAMPLE_POINTS = int(np.ceil(TOTAL_POINTS / SPLIT_SIZE))

def main():
    config = get_config('test')
    print(config.exp_dir)
    # create network and training agent
    tr_agent = get_agent(config)
    if config.ckpt:
        tr_agent.load_ckpt(config.ckpt)

    extra_pts = np.zeros((1, SPLIT_SIZE * NUM_SAMPLE_POINTS - TOTAL_POINTS, 3), dtype=np.float32)
    batch_points = np.zeros((SPLIT_SIZE, 0, NUM_SAMPLE_POINTS, 3), dtype=np.float32)
    num_sp_point = 6
    for b in range(config.batch_size):
        sdf_params = [-1.0,-1.0,-1.0,1.0,1.0,1.0]
        x_ = np.linspace(sdf_params[0], sdf_params[3], num=RESOLUTION)
        y_ = np.linspace(sdf_params[1], sdf_params[4], num=RESOLUTION)
        z_ = np.linspace(sdf_params[2], sdf_params[5], num=RESOLUTION)
        z, y, x = np.meshgrid(z_, y_, x_, indexing='ij')
        x = np.expand_dims(x, 3)
        y = np.expand_dims(y, 3)
        z = np.expand_dims(z, 3)
        all_pts = np.concatenate((x, y, z), axis=3).astype(np.float32)
        all_pts = all_pts.reshape(1, -1, 3)
        all_pts = np.concatenate((all_pts, extra_pts), axis=1).reshape(SPLIT_SIZE, 1, -1, 3)
        batch_points = np.concatenate((batch_points, all_pts), axis=1)

    pred_affs_all = np.zeros((SPLIT_SIZE, config.batch_size, NUM_SAMPLE_POINTS, 3*num_sp_point))

    for sp in range(SPLIT_SIZE):
        tr_agent.net.eval()
        with torch.no_grad():
            pred_affs = tr_agent.net.module.get_aff(torch.tensor(batch_points[sp]).cuda())
        pred_affs_all[sp, :, :, :] = pred_affs.detach().cpu().numpy()

    pred_affs_all = np.swapaxes(pred_affs_all, 0, 1)  # B, S, NUM SAMPLE, 1 or 2
    pred_affs_all = pred_affs_all.reshape((config.batch_size, -1, 3*num_sp_point))[:, :TOTAL_POINTS, :]

    batch_points = np.swapaxes(batch_points, 0, 1)  # B, S, NUM SAMPLE, 3
    batch_points = batch_points.reshape((config.batch_size, -1, 3))[:, :TOTAL_POINTS, :]

    fixed_affs_global = np.concatenate((
    np.concatenate((batch_points[:, :, 0:2], -batch_points[:, :, 2:3]), axis=2),
    np.concatenate((-batch_points[:, :, 0:1], batch_points[:, :, 1:3]), axis=2),
    np.concatenate((batch_points[:, :, 0:1], -batch_points[:, :, 1:2], batch_points[:, :, 2:3]), axis=2),
    np.concatenate((-batch_points[:, :, 0:2], batch_points[:, :, 2:3]), axis=2),
    np.concatenate((batch_points[:, :, 0:1], -batch_points[:, :, 1:3]), axis=2),
    np.concatenate((-batch_points[:, :, 0:1], batch_points[:, :, 1:2], -batch_points[:, :, 2:3]), axis=2)
    ), axis=2)

    fixed_affs_local = np.concatenate((
    np.concatenate((batch_points[:, :, 0:2], batch_points[:, :, 2:3]+0.1), axis=2),
    np.concatenate((batch_points[:, :, 0:1]+0.1, batch_points[:, :, 1:3]), axis=2),
    np.concatenate((batch_points[:, :, 0:1], batch_points[:, :, 1:2]+0.1, batch_points[:, :, 2:3]), axis=2),
    np.concatenate((batch_points[:, :, 0:2], batch_points[:, :, 2:3]-0.1), axis=2),
    np.concatenate((batch_points[:, :, 0:1]-0.1, batch_points[:, :, 1:3]), axis=2),
    np.concatenate((batch_points[:, :, 0:1], batch_points[:, :, 1:2]-0.1, batch_points[:, :, 2:3]), axis=2)
    ), axis=2)

    data = {"input":batch_points, "pred_affs":pred_affs_all,"fixed_affs_global":fixed_affs_global,"fixed_affs_local":fixed_affs_local}
    np.save("spatial_pattern" + "_" + str(config.ckpt) + ".npy", data)
    print('spatial pattern saved.')
main()

# python visualization/get_sp.py --category all --exp_name all --batch_size 1 --ckpt 30 -g 0