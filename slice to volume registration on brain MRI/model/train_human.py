from torch.utils.tensorboard import SummaryWriter
import os, glob
import sys
# from torch.utils.data import DataLoader
from tqdm import tqdm
# from data import datasets, trans
import numpy as np
import torch
# from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
import neurite as ne
import gc


# os.chdir('/content/drive/MyDrive/Guo_Lab/Registration/model/IC-VoxelMorpher')
import utils
import losses
from options.train_options import get_CycleTransMorph_config
from data.data_loader import CreateDataLoader
from models.CycleTransMorph import CycleTransMorph
from train_TransMorph import main


'''
GPU configuration
'''

GPU_iden = 0
GPU_num = torch.cuda.device_count()
print('Number of GPU: ' + str(GPU_num))
for GPU_idx in range(GPU_num):
    GPU_name = torch.cuda.get_device_name(GPU_idx)
    print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
torch.cuda.set_device(GPU_iden)
GPU_avai = torch.cuda.is_available()
print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
print('If the GPU is available? ' + str(GPU_avai))


config = get_CycleTransMorph_config(specie='Human')


main(config)