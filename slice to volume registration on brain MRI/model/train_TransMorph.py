from torch.nn.modules.activation import Tanh
from torch.utils.tensorboard import SummaryWriter
import os, utils, glob, losses
import sys
import math
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
import torchio as tio
import matplotlib.pyplot as plt
from options.train_options import get_CycleTransMorph_config
from data.data_loader import CreateDataLoader
from models.CycleTransMorph import CycleTransMorph
from data.data_util import *
from models.vgg3D_for_transformer import VGG_registration



class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir + "logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
    
def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=1.4):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)
        


def main(config):
    
    print('##############################################################')
    print("Configuration:")
    print(config)
    print('##############################################################')
    
    exp_path = config.root+'/model/experiment'
    save_dir = config.checkpoints_dir
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
        model = CycleTransMorph(config)
        # model = VGG_registration(config)
        model.to(config.device)
    else:
        model = CycleTransMorph(config)
        # model = VGG_registration(config)
        model.to(config.device)
    updated_lr = lr

    '''
    Initialize training
    '''
    ##############################################################################
    # Initialize dataloader
    ##############################################################################
    print("loading data")
    config.train=True
    data_loader_train = CreateDataLoader(config)
    dataset_train = data_loader_train.load_data()
    dataset_size_train = len(data_loader_train)
    
    config.train=False
    data_loader_valid = CreateDataLoader(config)
    dataset_valid = data_loader_valid.load_data()
    dataset_size_valid = len(data_loader_valid)
    print('#training images = %d' % dataset_size_train)
    
    data_loader_validation = CreateDataLoader(get_CycleTransMorph_config(type='validation', specie=config.specie))
    dataset_validation = data_loader_validation.load_data()
    dataset_size_validation = len(data_loader_validation)
    print('#training images = %d' % dataset_size_validation)
    print('Data Loaded!')

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    
    # PATH="/content/drive/MyDrive/Chenghao_CycleICTD/model/experiment/saved_model_CycleTransMorph/CheckPoint_26_model_SSIM.pt"
    # checkpoint = torch.load(PATH)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch_start = checkpoint['epoch']+1
    # loss = checkpoint['loss']


    if config.similarity_loss_type == 'NCC':
        criterion = losses.NCC_vxm()    
    elif config.similarity_loss_type == 'MSE':
        criterion = losses.MSE().loss
    elif config.similarity_loss_type == 'SSIM':
        criterion = losses.SSIM()
    elif config.similarity_loss_type == 'MI':
        criterion = losses.MutualInformation()
    elif config.similarity_loss_type == 'MS-SSIM':
        criterion = losses.MS_SSIM(data_range=1, channel=1, spatial_dims=3)

    criterion_param = nn.L1Loss()
    criterion_img=losses.SSIM()
    criterion_img_l1=nn.L1Loss()
    img_minus=True
    stop_num = 10
    early_stop = stop_num
    ##############################################################################
    # Learning curve
    ##############################################################################
    LossPath = os.path.join(exp_path + save_dir)
    print('LossPath',LossPath)
    with open(os.path.join(LossPath + '/learning_curve.csv'), 'a', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        # wr.writerow(['Current Epoch', 'Train Total Loss', 'Train Similarity {} Loss'.format(config.similarity_loss_type), 'Train IC {} Loss'.format(config.similarity_loss_type), 'Train SSIM', 'Train Pearson Corr', 'Val Total Loss', 'Val Similarity {} Loss'.format(config.similarity_loss_type), 'Val IC {} Loss'.format(config.similarity_loss_type), 'Val SSIM', 'Val Pearson Corr'])
        wr.writerow(['Current Epoch', 'param loss weight','Train Total Loss', 'Train Param Loss','Train image Similarity {} Loss'.format(config.similarity_loss_type),'train_pearson','train img l1',\
        'Val Total Loss', 'Val Param Loss','Val image Similarity {} Loss'.format(config.similarity_loss_type),'Val pearson','Val img L1'])
            
        f.close()
    
    print('Start training ...')
    torch.cuda.empty_cache()
    best_loss = np.inf
    param_loss_weight = 1
    img_loss_weight = 0
    ################################################################################################################################
    ########################################################### Training ###########################################################
    ################################################################################################################################
    for epoch in range(epoch_start, max_epoch):
        epoch_loss = []
        epoch_total_loss = []
        img_loss_list = []
        img_loss_l1_list=[]
        pearson_list = []
        param_loss_list = []
        epoch_step_time = []
        


        for step in tqdm(range(config.train_iteration)):#config.train_iteration

            step_start_time = time.time()
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            (data,param_gt) = next(iter(dataset_train.__iter__()))

            param_pre = model(data.to(config.device))
            # central_slice = get_central_slice(param_pre,filename,list(config.img_size),same_sub=config.same_sub,grad=True)
            # central_slice = get_central_slice(affine_param.detach(),filename,list(config.img_size),same_sub=config.same_sub)
            matrix_gt = get_affine(param_gt,config.img_size,config.same_sub,config.device)
            matrix_pre = get_affine(param_pre,config.img_size,config.same_sub,config.device)
            matrix_pre=matrix_pre[:,None,:,:]
            matrix_gt=matrix_gt[:,None,:,:]
            center_pre = get_central_slice(matrix_pre,data[:,0,:,:,:],config.device)
            center_gt = get_central_slice(matrix_gt.to(config.device),data[:,0,:,:,:],config.device)

            center_pre = center_pre.to(config.device).unsqueeze(1)
            center_gt = center_gt.to(config.device).unsqueeze(1)


            loss = 0
            loss_list = []
            
            param_loss = criterion_param(param_pre.to(torch.float32).to(config.device),param_gt.to(torch.float32).to(config.device))

            # img_loss = criterion_img(param_pre.to(torch.float32).to(config.device),gt_matrix.to(torch.float32).to(config.device))
            # img_loss = 1-criterion_img(slice_inp.to(torch.float32).to(config.device),central_slice.to(torch.float32).to(config.device))
            img_loss = 1-criterion_img(center_pre.to(torch.float32).to(config.device),center_gt.to(torch.float32).to(config.device))
            l1_img_loss = criterion_img_l1(center_pre.to(torch.float32).to(config.device),center_gt.to(torch.float32).to(config.device))
            pearson_corr,_ = pearsonr(center_pre.detach().cpu().numpy().squeeze().flatten(),center_gt.detach().cpu().numpy().squeeze().flatten())
            param_loss_list.append(param_loss)
            img_loss_list.append(img_loss)
            epoch_loss.append(loss)
            img_loss_l1_list.append(l1_img_loss)
            loss = param_loss_weight*param_loss + img_loss_weight*img_loss
            epoch_total_loss.append(loss.item())
            pearson_list.append(pearson_corr)
            # #compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step%50 == 0:
                grad_norms = [torch.norm(p.grad) for p in model.parameters()]
                print(f"Gradient norms at iteration {step}: {grad_norms}")
                print("param_loss",param_loss)
                print("img_loss",img_loss)
                print('loss',loss)
                # print("GT",matrix_gt)
                # print("pre",matrix_pre)

                center_pre_plot = center_pre.clone().detach().cpu().numpy()
                center_gt_plot = center_gt.clone().detach().cpu().numpy()
                plt.subplot(1,3,1)
                plt.imshow(center_pre_plot[0,0,:,:], cmap='gray')#GT
                plt.subplot(1,3,2)
                plt.imshow(center_gt_plot[0,0,:,:], cmap='gray')#prediction
                plt.subplot(1,3,3)
                plt.imshow(np.abs(center_pre_plot[0,0,:,:]-center_gt_plot[0,0,:,:]), cmap='gray')#prediction
                plt.show()
                
            # get compute time
            epoch_step_time.append(time.time() - step_start_time)
        train_param_loss = sum(param_loss_list) / len(param_loss_list)
        train_img_loss = sum(img_loss_list) / len(img_loss_list)
        train_img_l1_loss = sum(img_loss_l1_list) / len(img_loss_l1_list)
        train_loss = sum(epoch_total_loss) / len(epoch_total_loss)
        train_pearson = sum(pearson_list) / len(pearson_list)


        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{max_epoch:03d} ] loss = {train_loss:.5f}")
        # print(f"param_loss = {train_param_loss:.5f}")
        # print(f"train_img_loss = {train_img_loss:.5f}")

##################################################################################################################################
########################################################### Validation ###########################################################
##################################################################################################################################
        with torch.no_grad():
            model.eval()

            epoch_loss = []
            epoch_total_loss = []
            img_loss_list = []
            img_loss_l1_list=[]
            param_loss_list = []
            epoch_step_time = []
            pearson_list = []

            for step in tqdm(range(config.validation_iteration)):

                step_start_time = time.time()

                # generate inputs (and true outputs) and convert them to tensors
                adjust_learning_rate(optimizer, epoch, max_epoch, lr)
                (data,param_gt) = next(iter(dataset_validation.__iter__()))

                param_pre = model(data.to(config.device))
                # central_slice = get_central_slice(param_pre,filename,list(config.img_size),same_sub=config.same_sub,grad=True)
                # central_slice = get_central_slice(affine_param.detach(),filename,list(config.img_size),same_sub=config.same_sub)
                matrix_gt = get_affine(param_gt,config.img_size,config.same_sub,config.device)
                matrix_pre = get_affine(param_pre,config.img_size,config.same_sub,config.device)
                matrix_pre=matrix_pre[:,None,:,:]
                matrix_gt=matrix_gt[:,None,:,:]
                center_pre = get_central_slice(matrix_pre,data[:,0,:,:,:],config.device)
                center_gt = get_central_slice(matrix_gt.to(config.device),data[:,0,:,:,:],config.device)

                center_pre = center_pre.to(config.device).unsqueeze(1)
                center_gt = center_gt.to(config.device).unsqueeze(1)


                loss = 0
                loss_list = []
                
                param_loss = criterion_param(param_pre.to(torch.float32).to(config.device),param_gt.to(torch.float32).to(config.device))

                # img_loss = criterion_img(param_pre.to(torch.float32).to(config.device),gt_matrix.to(torch.float32).to(config.device))
                # img_loss = 1-criterion_img(slice_inp.to(torch.float32).to(config.device),central_slice.to(torch.float32).to(config.device))
                img_loss = 1-criterion_img(center_pre.to(torch.float32).to(config.device),center_gt.to(torch.float32).to(config.device))
                l1_img_loss = criterion_img_l1(center_pre.to(torch.float32).to(config.device),center_gt.to(torch.float32).to(config.device))
                pearson_corr,_ = pearsonr(center_pre.detach().cpu().numpy().squeeze().flatten(),center_gt.detach().cpu().numpy().squeeze().flatten())
                param_loss_list.append(param_loss)
                img_loss_list.append(img_loss)
                img_loss_l1_list.append(l1_img_loss)
                loss = param_loss_weight*param_loss + img_loss_weight*img_loss
                epoch_loss.append(loss)
                epoch_total_loss.append(loss.item())
                pearson_list.append(pearson_corr)
                # if step%50 == 0:
                #     grad_norms = [torch.norm(p.grad) for p in model.parameters()]
                #     print(f"Gradient norms at iteration {step}: {grad_norms}")
                #     print("param_loss",param_loss)
                #     print("img_loss",img_loss)
                #     print("GT",param_GT)
                #     print("pre",param_pre)


                # get compute time
                epoch_step_time.append(time.time() - step_start_time)

            valid_param_loss = sum(param_loss_list) / len(param_loss_list)
            valid_img_loss = sum(img_loss_list) / len(img_loss_list)
            valid_img_l1_loss = sum(img_loss_l1_list) / len(img_loss_l1_list)
            valid_loss = sum(epoch_total_loss) / len(epoch_total_loss)
            valid_pearson = sum(pearson_list) / len(pearson_list)
            # Print the information.
            print(f"[ Valid | {epoch + 1:03d}/{max_epoch:03d} ] loss = {valid_loss:.5f}")
            # print(f"param_loss = {valid_param_loss:.5f}")
            # print(f"img_loss = {valid_img_loss:.5f}")


        ##############################################################################
        # Learning curve
        ##############################################################################
        with open(os.path.join(LossPath + '/learning_curve.csv'), 'a', encoding='utf-8', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['%d' %(epoch + 1), '%.15f' % param_loss_weight, '%.15f' % train_loss, '%.15f' % train_param_loss, '%.15f' % train_img_loss, '%.15f' % train_pearson,'%.15f' % train_img_l1_loss,\
                        '%.15f' % valid_loss, '%.15f' % valid_param_loss, '%.15f' % valid_img_loss,'%.15f' % valid_pearson,'%.15f' % valid_img_l1_loss])
            f.close()
            

        #############################################################################
        # Save best model and early stop
        #############################################################################

        if valid_loss < best_loss:
            best_epoch = epoch
            best_loss = valid_loss
            model.save(os.path.join(LossPath, 'Best_model_{}.pt'.format(config.similarity_loss_type)))
            early_stop = stop_num
        else:
            early_stop = early_stop - 1
        if epoch % 10 == 1:    ### update
            model.save(os.path.join(LossPath, 'Best_{}_model_{}.pt'.format(epoch, config.similarity_loss_type)))  
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            }, os.path.join(LossPath, 'CheckPoint_{}_model_{}.pt'.format(epoch, config.similarity_loss_type)))
        if early_stop and epoch < max_epoch - 1:  
            continue  
        else:
          print('early stop at'+str(epoch)+'epoch')
          break  
        model.save(os.path.join(LossPath, 'model_{}.pt'.format(config.similarity_loss_type)))  
        # if max_pearson < np.mean(step_pearson):
        #     best_epoch = epoch
        #     max_pearson = np.mean(step_pearson)
        #     model.save(os.path.join(LossPath, 'Best_model_{}.pt'.format(config.similarity_loss_type)))
        # if max_pearson < np.mean(validation_average_epoch_pearson[-10:]): 
        #     # best_epoch = epoch        
        #     # max_pearson = np.mean(step_pearson)
        #     # model.save(os.path.join(LossPath, 'Best_model_{}.pt'.format(config.similarity_loss_type)))       
        #     early_stop = stop_num
        # else:
        #     early_stop = early_stop - 1

        # if epoch % 5 == 1:    ### update
        #     model.save(os.path.join(LossPath, 'Best_{}_model_{}.pt'.format(epoch, config.similarity_loss_type)))    
        
        # if early_stop and epoch < max_epoch - 1:    
        #     continue    


# if __name__ == '__main__':
#     '''
#     GPU configuration
#     '''
#     GPU_iden = 0
#     GPU_num = torch.cuda.device_count()
#     print('Number of GPU: ' + str(GPU_num))
#     for GPU_idx in range(GPU_num):
#         GPU_name = torch.cuda.get_device_name(GPU_idx)
#         print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
#     torch.cuda.set_device(GPU_iden)
#     GPU_avai = torch.cuda.is_available()
#     print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
#     print('If the GPU is available? ' + str(GPU_avai))
#     main()
