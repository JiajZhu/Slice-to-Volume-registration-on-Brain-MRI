from torch.utils.tensorboard import SummaryWriter
import os, glob
import sys
# from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
# from data import datasets, trans
import numpy as np
import torch
# from torchvision import transforms
from torch import optim
import torch.nn as nn
from scipy.stats import pearsonr
# from natsort import natsorted
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, '/media/sail/HDD18T/BME_Grad_Project/Chenghao_CycleICTD/model')
import losses
from options.inference_options import get_CycleTransMorph_config
from data.data_loader import CreateDataLoader
from models.CycleTransMorph import CycleTransMorph



class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir + "logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
    
def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)
        


def main(config):
    
    print('##############################################################')
    print("Configuration:")
    print(config)
    print('##############################################################')
    
    exp_path = '/media/sail/HDD18T/BME_Grad_Project/Chenghao_CycleICTD/model/experiment'
    save_dir = config.checkpoints_dir
    
    print(exp_path + save_dir + config.checkpoint)

    if not os.path.exists(exp_path + save_dir):
        os.makedirs(exp_path + save_dir)

    sys.stdout = Logger(exp_path + save_dir)
    
    lr = config.lr # learning rate
    epoch_start = 0
    max_epoch = config.max_epoch #max traning epoch
    cont_training = config.use_checkpoint #if continue training

    '''
    If continue from previous training
    '''
    if cont_training:
        print('Using checkpoint: ', config.checkpoint)
        model = CycleTransMorph(config).load(exp_path + save_dir + config.checkpoint, config.device)
        model.to(config.device)
    else:
        model = CycleTransMorph(config)
        model.to(config.device)
    updated_lr = lr

    '''
    Initialize training
    '''
    ##############################################################################
    # Initialize dataloader
    ##############################################################################

    # data_loader_train = CreateDataLoader(config)
    # dataset_train = data_loader_train.load_data()
    # dataset_size_train = len(data_loader_train)
    # print('#training images = %d' % dataset_size_train)
    
    data_loader_test = CreateDataLoader(get_CycleTransMorph_config(type='test', specie=config.specie))
    dataset_test = data_loader_test.load_data()
    dataset_size_test = len(data_loader_test)
    print('#test images = %d' % dataset_size_test)
    print('Data Loaded!')

    
    # Evaluation metric
    ssim = losses.SSIM3D()
    
    ##############################################################################
    # Metrics
    ##############################################################################
    ssim = losses.SSIM3D()
    max_pearson = 0
    stop_num = 10
    early_stop = stop_num
    
    ##############################################################################
    # Learning curve
    ##############################################################################
    LossPath = os.path.join(exp_path + save_dir)
    
    print('Start testing ...')
    torch.cuda.empty_cache()
    

##################################################################################################################################
########################################################### test #################################################################
##################################################################################################################################
    with torch.no_grad():
        model.eval()

        epoch_loss = []
        epoch_total_loss = []
        epoch_step_time = []
        step_ssim = []
        step_pearson = []

        for step in tqdm(range(config.validation_iteration)):

            step_start_time = time.time()

            # generate inputs (and true outputs) and convert them to tensors
            data = next(iter(dataset_test.__iter__()))
            moving = data[0][0].permute((0, 4, 1, 2, 3)).cuda()
            fixed = data[0][1].permute((0, 4, 1, 2, 3)).cuda()

            out_mf, out_fm = model(moving, fixed)
            # out_mf = model(moving, fixed)
            # out_fm = model(fixed, moving)
        

            input_moving = moving.permute(0, 2, 3, 4, 1).detach().cpu().numpy().astype(np.float32).squeeze()
            input_fixed = fixed.permute(0, 2, 3, 4, 1).detach().cpu().numpy().astype(np.float32).squeeze()

            moved_mf1 = out_mf[0].permute(0, 2, 3, 4, 1).detach().cpu().numpy().astype(np.float32).squeeze()
            warp_mf1 = out_mf[1].permute(0, 2, 3, 4, 1).detach().cpu().numpy().astype(np.float32).squeeze()
            
            print(input_moving.shape)
            print(input_fixed.shape)
            print(moved_mf1.shape)
            print(warp_mf1.shape)

            # moved_fm1 = out_fm[0].permute(0, 2, 3, 4, 1).detach().cpu().numpy().astype(np.float32)
            # warp_fm1 = out_fm[1].permute(0, 2, 3, 4, 1).detach().cpu().numpy().astype(np.float32)
            
            
        # from utils import jacobian_determinant_vxm
        # jacobian_det_np_arr = jacobian_determinant_vxm(warp_mf[0, ...])

            # Evaluation metrics
            ssim_val = 0
            pearson_val = 0
            # for n in range(2):
            for i in range(out_mf[0].shape[0]):
                ssim_val += 1 - ssim(torch.unsqueeze(out_mf[0][i, ...], 0), torch.unsqueeze(fixed[i, ...], 0)).item()
                ssim_val += 1 - ssim(torch.unsqueeze(out_fm[0][i, ...], 0), torch.unsqueeze(moving[i, ...], 0)).item()
                pearson_val += pearsonr(out_mf[0][i, ...].detach().cpu().numpy().squeeze().flatten(), fixed[i, ...].detach().cpu().numpy().squeeze().flatten())[0]
                pearson_val += pearsonr(out_fm[0][i, ...].detach().cpu().numpy().squeeze().flatten(), moving[i, ...].detach().cpu().numpy().squeeze().flatten())[0]
            
            step_ssim.append(ssim_val / out_mf[0].shape[0] / 2)
            step_pearson.append(pearson_val / out_mf[0].shape[0] / 2)

            # get compute time
            epoch_step_time.append(time.time() - step_start_time)
            
            fig = plt.figure(figsize=(20, 12))

            rows = 3
            cols = 4
            count = 1

            slices1 = 80
            slices2 = 96
            slices3 = 96

            result = [input_moving, input_fixed, moved_mf1, warp_mf1]

            for img in result:
                
                if count % 4 == 0:
                    i = (img - np.min(img)) / (np.max(img) - np.min(img))
                    cmap = None
                else:
                    i = img[...]
                    cmap = 'gray'
                    
                plt.subplot(rows, cols, count)
                plt.imshow(np.rot90(i[slices1, :, :]), cmap=cmap)
                plt.clim(0, 1)
                plt.axis('off')
                if count <= 4:
                    plt.title('$I_M$')
                
                plt.subplot(rows, cols, count + len(result))
                plt.imshow(np.rot90(i[:, slices2, :]), cmap=cmap)
                plt.clim(0, 1)
                plt.axis('off')
                if count <= 4:
                    plt.title('$I_F$')
                
                plt.subplot(rows, cols, count + len(result) * 2)
                plt.imshow(np.rot90(i[:, :, slices3]), cmap=cmap)
                plt.clim(0, 1)
                plt.axis('off')
                if count <= 4:
                    plt.title('$I_{MF}$')
                
                count += 1
            # Save the full figure...
            fig.savefig('full_figure.png')

    print('SSIM: ', step_ssim)
    print('PCC: ', step_pearson)