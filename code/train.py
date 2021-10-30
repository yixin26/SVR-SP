from collections import OrderedDict
from tqdm import tqdm
from utils import cycle
import torch
from dataset import get_dataloader
from config import get_config
from agent import get_agent

def main():
    torch.backends.cudnn.enabled = False

    # create experiment config
    config = get_config('train')
    # create network and training agent
    tr_agent = get_agent(config)

    # load from checkpoint if provided
    if config.cont:
        tr_agent.load_ckpt(config.ckpt)

    # create dataloader
    train_loader = get_dataloader('train', config)

    # start training
    clock = tr_agent.clock
    for e in range(clock.epoch, config.nr_epochs):
        train_loader.dataset.resample_data()
        pbar = tqdm(train_loader)
        tot_sdf_loss,tot_accuracy,tot_sdf_loss_realvalue, tot_regularization_loss = 0,0,0,0
        num = 0
        for b, data in enumerate(pbar):
            outputs, losses = tr_agent.train_func(data)
            pbar.set_description("EPOCH[{}][{}]".format(e, b))
            pbar.set_postfix(OrderedDict({k: '%.5f' % v.item() for k, v in losses.items()}))

            tot_sdf_loss +=losses["sdf_loss"]
            tot_accuracy +=losses["accuracy"]
            tot_sdf_loss_realvalue +=losses["sdf_loss_realvalue"]
            tot_regularization_loss +=losses["regularization_loss"]

            num += 1
            clock.tick()

        print('tot_sdf_loss={}'.format(tot_sdf_loss/num))
        print('tot_accuracy={}'.format(tot_accuracy/num))
        print('tot_sdf_loss_realvalue={}'.format(tot_sdf_loss_realvalue/num))
        print('tot_regularization_loss={}'.format(tot_regularization_loss/num))

        with open(config.exp_name+'_log.txt', 'a') as logfile:
            logtt = 'Epoch {}/{} - lr: {} - tot_sdf_loss: {} - tot_accuracy: {} - tot_sdf_loss_realvalue: {} \n'.format(
                e, config.nr_epochs, tr_agent.optimizer.param_groups[-1]['lr'], tot_sdf_loss/num,tot_accuracy/num, tot_sdf_loss_realvalue/num)
            logfile.write(logtt)

        # update lr
        tr_agent.update_learning_rate()
        print('update learning rate={}'.format(tr_agent.optimizer.param_groups[-1]['lr']))

        if clock.epoch % config.save_frequency == 0:
            tr_agent.save_ckpt()

        clock.tock()
        tr_agent.save_ckpt('latest')

if __name__ == '__main__':
    main()

#python train.py --category all --exp all -g 0,1