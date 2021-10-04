import os
import json
import argparse
import shutil
from utils import ensure_dirs

def get_config(phase):
    config = Config(phase)
    return config

class Config(object):

    def __init__(self, phase):
        self.is_train = phase == 'train'

        # init hyperparameters and parse from command-line
        parser, args = self.parse()
        self.phase = phase

        print("-----Experiment Configuration-----")
        for k,v in args.__dict__.items():
            print(k,v)
            self.__setattr__(k, v)

        #self.exp_name = 'exp_{}'.format(self.category)

        self.exp_dir = os.path.join(self.proj_dir, self.exp_name)
        if phase == 'train' and args.cont is not True and os.path.exists(self.exp_dir):
            response = input('Experiment log/model already exists, overwrite? (y/n)')
            if response != 'y':
                exit()
            shutil.rmtree(self.exp_dir)

        self.log_dir = os.path.join(self.exp_dir, 'log')
        self.model_dir = os.path.join(self.exp_dir, 'model')
        ensure_dirs([self.log_dir, self.model_dir])

        # GPU usage
        self.parallel = False
        self.use_gpu = False
        if args.gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
            self.use_gpu = True
            if len(str(args.gpu_ids).split(',')) > 1:
                self.parallel = True
        print('use gpu:',self.use_gpu)
        if self.is_train:
            with open(os.path.join(self.log_dir, 'config.txt'), 'w') as f:
                json.dump(args.__dict__, f, indent=2)

    def parse(self):
        parser = argparse.ArgumentParser()
        self._add_basic_config_(parser)
        self._add_dataset_config_(parser)
        self._add_network_config_(parser)
        self._add_training_config_(parser)
        args = parser.parse_args()
        return parser, args

    def _add_basic_config_(self, parser):
        group = parser.add_argument_group('basic')
        group.add_argument('--proj_dir', type=str, default='./', help='where models saved')
        group.add_argument('--exp_name', type=str, default='test', help='experiment name')
        group.add_argument('-g', '--gpu_ids', type=str, default=None, help='gpu to use, e.g. 0  0,1,2.')
        parser.add_argument('--model', type=str, default='SDFNet',choices=('SDFNet', 'CamNet'),help='Model to use [\'SDFNet, CamNet\']')

    def _add_dataset_config_(self, parser):
        group = parser.add_argument_group('dataset')
        group.add_argument('--batch_size', type=int, default=20, help='batch size')
        group.add_argument('--category', type=str, default='chair', help='shape category name')
        group.add_argument('--num_workers', type=int, default=8, help='number of workers for data loading')
        group.add_argument('--sdf_h5_path', type=str, default='/mnt/disk7/yixin/data/ShapeNet/SDF_v1', help='where sdf data is')
        #group.add_argument('--render_h5_path', type=str, default='/mnt/disk7/yixin/data/ShapeNet/ShapeNetRenderingh5_v1', help='where render data is')
        group.add_argument('--render_h5_path', type=str, default='/mnt/disk7/yixin/data/ShapeNet/PretrainedEsth5', help='where render data is')

        group.add_argument('--id_path', type=str, default='/mnt/disk7/yixin/data/ShapeNet/filelists')
        group.add_argument('--mesh_obj_path', type=str, default='/mnt/disk7/yixin/data/ShapeNet/march_cube_objs_v1', help='where render data is')
        group.add_argument('--img_h', type=int, default=137)
        group.add_argument('--img_w', type=int, default=137)
        group.add_argument('--num_sample_points', type=int, default=2048)
        group.add_argument('--cat_limit', type=int, default=36000, help="balance each category, 1500 * 24 = 36000")

    def _add_network_config_(self, parser):
        group = parser.add_argument_group('network')
        group.add_argument('--sdf_weight', type=float, default=10.)
        group.add_argument('--mask_weight', type=float, default=4.0)
        group.add_argument('--delta', type=float, default=0.01)
        group.add_argument('--use_tanh', action='store_true')

    def _add_training_config_(self, parser):
        group = parser.add_argument_group('training')
        group.add_argument('--nr_epochs', type=int, default=50, help='total number of epochs to train')
        group.add_argument('--wd', action='store_true', default=False, help='weight decay')
        group.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
        group.add_argument('--lr_step_size', type=int, default=5, help='step size for learning rate decay')
        group.add_argument('--lr_decay', type=float, default=0.9, help='factor for lr decay')

        group.add_argument('--continue', dest='cont', action='store_true', help='continue training from checkpoint')
        group.add_argument('--ckpt', type=str, default='latest', required=False, help='desired checkpoint to restore')
        group.add_argument('--reset_lr', dest='reset_lr', action='store_true', help='reset your learning rate')
        group.add_argument('--vis', action='store_true', default=False, help='visualize output in training')
        group.add_argument('--save_frequency', type=int, default=5, help='save models every x epochs')
        group.add_argument('--val_frequency', type=int, default=10, help='run validation every x iterations')
        group.add_argument('--vis_frequency', type=int, default=100, help='visualize output every x iterations')

        group.add_argument('--val', type=bool, default=False, help='with validation or not')

