from collections import OrderedDict
from tqdm import tqdm
from dataset import get_dataloader
from config import get_config
from agent import get_agent
import os
import numpy as np
import random
from joblib import Parallel, delayed
import pymesh
import torch
import struct
import sys
RESOLUTION = 65
TOTAL_POINTS = RESOLUTION * RESOLUTION * RESOLUTION
SPLIT_SIZE = int(np.ceil(TOTAL_POINTS / 24576.0 ))
NUM_SAMPLE_POINTS = int(np.ceil(TOTAL_POINTS / SPLIT_SIZE))

BASE_DIR = '/mnt/disk7/yixin/DISN'
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'data'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'preprocessing'))

def to_binary(res, pos, pred_sdf_val_all, sdf_file):
    f_sdf_bin = open(sdf_file, 'wb')

    f_sdf_bin.write(struct.pack('i', -res))  # write an int
    f_sdf_bin.write(struct.pack('i', res))  # write an int
    f_sdf_bin.write(struct.pack('i', res))  # write an int

    positions = struct.pack('d' * len(pos), *pos)
    f_sdf_bin.write(positions)
    val = struct.pack('=%sf'%pred_sdf_val_all.shape[0], *(pred_sdf_val_all))
    f_sdf_bin.write(val)
    f_sdf_bin.close()

def create_obj(pred_sdf_val, sdf_params, dir, cat_id, obj_nm, view_id, i):
    if not isinstance(view_id, str):
        view_id = "%02d"%view_id
    dir = os.path.join(dir, cat_id)
    os.makedirs(dir, exist_ok=True)
    obj_nm = cat_id +"_" + obj_nm
    cube_obj_file = os.path.join(dir, obj_nm+"_"+view_id+".obj")
    sdf_file = os.path.join(dir, obj_nm+"_"+view_id+".dist")
    to_binary((RESOLUTION-1), sdf_params, pred_sdf_val, sdf_file)
    create_one_cube_obj("/mnt/disk7/yixin/DISN/isosurface/computeMarchingCubes", i, sdf_file, cube_obj_file)
    command_str = "rm -rf " + sdf_file
    #print("command:", command_str)
    os.system(command_str)

def create_one_cube_obj(marching_cube_command, i, sdf_file, cube_obj_file):
    command_str = marching_cube_command + " " + sdf_file + " " + cube_obj_file + " -i " + str(i)
    os.system(command_str)
    return cube_obj_file

def main():
    config = get_config('test')
    # create network and  agent
    tr_agent = get_agent(config)

    if config.ckpt:
        tr_agent.load_ckpt(config.ckpt)

    # create dataloader
    test_loader = get_dataloader('test', config)

    iso = 0.0
    RESULT_OBJ_PATH = os.path.join(config.exp_dir, 'results', str(config.ckpt),'test_objs', str(RESOLUTION) + "_" + str(iso))
    if not os.path.exists(RESULT_OBJ_PATH): os.makedirs(RESULT_OBJ_PATH, exist_ok=True)

    pbar = tqdm(test_loader)
    for e, batch_data in enumerate(pbar):
        extra_pts = np.zeros((1, SPLIT_SIZE * NUM_SAMPLE_POINTS - TOTAL_POINTS, 3), dtype=np.float32)
        batch_points = np.zeros((SPLIT_SIZE, 0, NUM_SAMPLE_POINTS, 3), dtype=np.float32)
        for b in range(config.batch_size):
            sdf_params = batch_data['sdf_params'][b]
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

        pred_sdf_val_all = np.zeros((SPLIT_SIZE, config.batch_size, NUM_SAMPLE_POINTS, 1))

        for sp in range(SPLIT_SIZE):
            data = batch_data
            data['sdf_pt'] = torch.tensor(batch_points[sp])
            pred_sdf_val = tr_agent.infer_func(data)
            pred_sdf_val_all[sp, :, :, :] = pred_sdf_val.detach().cpu().unsqueeze(2).numpy()

        pred_sdf_val_all = np.swapaxes(pred_sdf_val_all, 0, 1)  # B, S, NUM SAMPLE, 1 or 2
        pred_sdf_val_all = pred_sdf_val_all.reshape((config.batch_size, -1, 1))[:, :TOTAL_POINTS, :]

        result = pred_sdf_val_all / config.sdf_weight

        for b in range(config.batch_size):
            create_obj(result[b], batch_data['sdf_params'][b], RESULT_OBJ_PATH,
                            batch_data['cat_id'][b], batch_data['obj_nm'][b], batch_data['view_id'][b], iso)

if __name__ == '__main__':
    main()
