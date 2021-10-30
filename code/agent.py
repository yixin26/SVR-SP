import os
import torch
import torch.optim as optim
import torch.nn as nn
from abc import abstractmethod
from tensorboardX import SummaryWriter
from utils import TrainClock

class BaseAgent(object):
    """Base trainer that provides commom training behavior.
        All trainer should be subclass of this class.
    """

    def __init__(self, config):
        self.config = config
        self.log_dir = config.log_dir
        self.model_dir = config.model_dir
        self.clock = TrainClock()
        self.batch_size = config.batch_size

        # build network
        self.net = self.build_net(config)

        # set loss function
        self.set_loss_function()

        # set optimizer
        self.set_optimizer(config)

        # set tensorboard writer
        self.train_tb = SummaryWriter(os.path.join(self.log_dir, 'train.events'))
        self.val_tb = SummaryWriter(os.path.join(self.log_dir, 'val.events'))

    @abstractmethod
    def build_net(self, config):
        raise NotImplementedError

    def set_loss_function(self):
        """set loss function used in training"""
        self.criterion = nn.MSELoss().cuda()

    def set_optimizer(self, config):
        """set optimizer and lr scheduler used in training"""
        self.optimizer = optim.Adam(self.net.parameters(), config.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, config.lr_step_size, config.lr_decay)

    def save_ckpt(self, name=None):
        """save checkpoint during training for future restore"""
        if name is None:
            save_path = os.path.join(self.model_dir, "ckpt_epoch{}.pth".format(self.clock.epoch))
            print("Checkpoint saved at {}".format(save_path))
        else:
            save_path = os.path.join(self.model_dir, "{}.pth".format(name))
        if isinstance(self.net, nn.DataParallel):
            torch.save({
                'clock': self.clock.make_checkpoint(),
                'model_state_dict': self.net.module.cpu().state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            }, save_path)
        else:
            torch.save({
                'clock': self.clock.make_checkpoint(),
                'model_state_dict': self.net.cpu().state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            }, save_path)
        self.net.cuda()

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
        if ('optimizer_state_dict' in checkpoint.keys() and 'scheduler_state_dict' in checkpoint.keys() and 'clock' in checkpoint.keys()):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.clock.restore_checkpoint(checkpoint['clock'])

    @abstractmethod
    def forward(self, data):
        pass

    def update_network(self, loss_dict):
        """update network by back propagation"""
        loss = sum(loss_dict.values())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_learning_rate(self):
        """record and update learning rate"""
        self.train_tb.add_scalar('learning_rate', self.optimizer.param_groups[-1]['lr'], self.clock.epoch)
        self.scheduler.step(self.clock.epoch)

    def record_losses(self, loss_dict, mode='train'):
        losses_values = {k: v.item() for k, v in loss_dict.items()}

        # record loss to tensorboard
        tb = self.train_tb if mode == 'train' else self.val_tb
        for k, v in losses_values.items():
            tb.add_scalar(k, v, self.clock.step)

    def train_func(self, data):
        """one step of training"""
        self.net.train()

        outputs, losses = self.forward(data)

        if self.config.model == 'SDFNet':
            loss = {}
            loss['sdf_loss'] = losses['sdf_loss'] + losses['regularization_loss']
        else:
            raise ValueError

        self.update_network(loss)
        self.record_losses(losses, 'train')

        return outputs, losses

    def infer_func(self, data):
        self.net.eval()
        return self.infering(data)

    def val_func(self, data):
        """one step of validation"""
        self.net.eval()

        with torch.no_grad():
            outputs, losses = self.forward(data)

        self.record_losses(losses, 'validation')

        return outputs, losses

    def visualize_batch(self, data, mode, **kwargs):
        """write visualization results to tensorboard writer"""
        raise NotImplementedError

from torch.autograd import Function
import sampling_cuda

class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance
        :param ctx:
        :param xyz: (B, N, 3) where N > npoint
        :param npoint: int, number of features in the sampled set
        :return:
             output: (B, npoint) tensor containing the set
        """
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        sampling_cuda.furthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None
furthest_point_sample = FurthestPointSampling.apply

from sdfnet import get_model
def get_agent(config):
    return Agent(config)

class Agent(BaseAgent):

    def __init__(self, config):
        super(Agent, self).__init__(config)
        self.mask_weight = config.mask_weight
        self.sdf_weight = config.sdf_weight
        self.delta = config.delta

        self.set_loss_function()

    def build_net(self, config):
        net = get_model(config)
        if config.parallel:
            net = nn.DataParallel(net)
        if config.use_gpu:
            net = net.cuda()
        return net

    def eval(self, gt_sdf, pred_sdf):
        zero = 0.0
        gt_sign = gt_sdf > zero
        pred_sign = pred_sdf > zero
        accuracy = torch.mean((gt_sign==pred_sign).float())
        sdf_loss_realvalue = torch.mean(torch.abs(gt_sdf - pred_sdf / self.sdf_weight))
        return accuracy.detach(),sdf_loss_realvalue.detach()

    def forward(self, data):
        sample_pt = data['sdf_pt'].contiguous()
        target_sdf = data['sdf_val'].contiguous()
        img = data['img'].transpose(2, 3).transpose(1, 2).contiguous()
        trans_mat = data['trans_mat'].contiguous()

        if self.config.use_gpu:
            sample_pt = sample_pt.cuda()
            target_sdf = target_sdf.cuda()
            img = img.cuda()
            trans_mat = trans_mat.cuda()

        choice = furthest_point_sample(sample_pt, 2048).to(torch.int64)
        sample_pt = torch.gather(sample_pt, dim=1, index=choice.unsqueeze(2).expand(-1, -1, 3))
        target_sdf = torch.gather(target_sdf, dim=1, index=choice)

        output_sdf = self.net(img, sample_pt, trans_mat)

        weight_mask = (target_sdf<=self.delta) * self.mask_weight + (target_sdf>self.delta)
        sdf_loss = torch.mean(torch.abs(target_sdf * self.sdf_weight - output_sdf) * weight_mask) * 1000

        #add regularization
        regularization_loss = 0
        lamda = 1e-6
        for name, param in self.net.named_parameters():
            if 'bias' not in name:
                regularization_loss += torch.sum(abs(param))
        regularization_loss *= lamda

        accuracy, sdf_loss_realvalue = self.eval(target_sdf,output_sdf)

        return output_sdf, {"sdf_loss": sdf_loss,
                            "accuracy": accuracy,
                            "sdf_loss_realvalue": sdf_loss_realvalue,
                            "regularization_loss": regularization_loss
                            }

    def infering(self, data):
        sample_pt = data['sdf_pt'].contiguous()
        img = data['img'].transpose(2, 3).transpose(1, 2).contiguous()
        trans_mat = data['trans_mat'].contiguous()
        if self.config.use_gpu:
            sample_pt = sample_pt.cuda()
            img = img.cuda()
            trans_mat = trans_mat.cuda()

        output_sdf = self.net(img, sample_pt, trans_mat)

        return output_sdf