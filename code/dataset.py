from torch.utils.data import DataLoader, Dataset
import torch
import h5py
import numpy as np
import os
import random
import json
import copy

def get_dataloader(phase, config):
    is_shuffle = phase == 'train'

    dataset = SDFdata(phase, config, shuffle=is_shuffle)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=is_shuffle, num_workers=config.num_workers,
                            worker_init_fn=np.random.seed())
    return dataloader

def get_all_info(CUR_PATH):
    with open(CUR_PATH+'/info.json') as json_file:
        data = json.load(json_file)
        lst_dir, cats, all_cats, raw_dirs = data["lst_dir"], data['cats'], data['all_cats'], data["raw_dirs_v1"]
    return lst_dir, cats, all_cats, raw_dirs

def getids(info_dir,sdf_h5_path,cat_id,num_views):
    ids = []
    with open(info_dir, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            sdf_path = os.path.join(sdf_h5_path, cat_id, line.strip(), 'ori_sample.h5')
            if os.path.exists(sdf_path):
                for render in range(num_views):
                    ids += [(cat_id, line.strip(), render)]
    return ids

class SDFdata(Dataset):
    def __init__(self, phase, config, shuffle):
        self.config = config
        self.shuffle = shuffle

        self.sdf_h5_path = config.sdf_h5_path
        self.render_h5_path = config.render_h5_path
        self.mesh_obj_path = config.mesh_obj_path
        self.id_path = config.id_path
        self.categorys = config.category.split(',')
        self.gen_num_pt = config.num_sample_points

        lst_dir, cats, all_cats, raw_dirs = get_all_info(self.id_path)
        if 'all' in self.categorys:
            self.categorys = cats
        else:
            used_categorys = {}
            for c in self.categorys:
                used_categorys[c] = cats[c]
            self.categorys = used_categorys

        self.views = 24
        self.cats_limit = {}
        self.epoch_amount = 0
        self.ids = []
        for cat_name, cat_id in self.categorys.items():
            cat_list = os.path.join(self.id_path, cat_id + '_' + phase + '.lst')
            idlist = getids(cat_list,self.sdf_h5_path,cat_id,self.views)
            self.ids += idlist
            if phase == 'train':
                self.cats_limit[cat_id] = min(len(idlist), config.cat_limit)
            else:
                self.cats_limit[cat_id] = len(idlist)
            self.epoch_amount += self.cats_limit[cat_id]

        self.data_order = list(range(len(self.ids)))
        self.order = self.data_order #self.order would be changed in each iteration
        print('num of ',phase,' data:',self.epoch_amount)

    def __len__(self):
        return self.epoch_amount

    def resample_data(self):
        if self.shuffle:
            self.order = self.refill_data_order()
            print("data order reordered!")

    def refill_data_order(self):
        temp_order = copy.deepcopy(self.data_order)
        cats_quota = {key: value for key, value in self.cats_limit.items()}
        np.random.shuffle(temp_order)
        pointer = 0
        epoch_order=[]
        while len(epoch_order) < self.epoch_amount:
            cat_id, _, _ = self.ids[temp_order[pointer]]
            if cats_quota[cat_id] > 0:
                epoch_order.append(temp_order[pointer])
                cats_quota[cat_id]-=1
            pointer+=1
        return epoch_order

    def get_sdf_h5(self, sdf_h5_file, cat_id, obj):
        h5_f = h5py.File(sdf_h5_file, 'r')
        try:
            if ('pc_sdf_sample' in h5_f.keys()
                    and 'sdf_params' in h5_f.keys()):
                sample_sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
                if sample_sdf.shape[1] == 4:
                    sample_pt, sample_sdf_val = sample_sdf[:, :3], sample_sdf[:, 3]
                else:
                    sample_pt, sample_sdf_val = None, sample_sdf[:, 0]
                sdf_params = h5_f['sdf_params'][:]
            else:
                raise Exception(cat_id, obj, "no sdf and sample")
        finally:
            h5_f.close()
        return sample_pt, sample_sdf_val, sdf_params

    def get_img(self, img_h5):
        with h5py.File(img_h5, 'r') as h5_f:
            trans_mat = h5_f["trans_mat"][:].astype(np.float32)
            img_raw = h5_f["img_arr"][:]
            img_arr = img_raw[:, :, :3]
            img_arr = np.clip(img_arr, 0, 255)
            img_arr = img_arr.astype(np.float32) / 255.

            return img_arr, trans_mat

    def __getitem__(self, index):
        cat_id, sdf_name, view = self.ids[self.order[index]]
        sdf_path = os.path.join(self.sdf_h5_path, cat_id, sdf_name, 'ori_sample.h5')
        render_path = os.path.join(self.render_h5_path, cat_id, sdf_name, "%02d.h5" % view)

        sample_pt, sample_sdf_val, sdf_params = self.get_sdf_h5(sdf_path, cat_id, sdf_name)
        img, trans_mat = self.get_img(render_path)

        return {'sdf_pt':sample_pt,
                'sdf_val':sample_sdf_val,
                'sdf_params':sdf_params,
                'img': img, #HxWx4 (137x137)
                'trans_mat': trans_mat, #3x4
                'cat_id':cat_id,
                'view_id':view,
                'obj_nm':render_path.split('/')[-2]
                }
