### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
### This script is modified based on the pix2pixHD official implementation (see license above)
### https://github.com/NVIDIA/pix2pixHD

import os.path
from .base_dataset import BaseDataset
from data.data_util import *
import torch
import torchio as tio
# import nibabel as nib
import numpy as np
import nibabel as nib
# import random

class S2VDataset(BaseDataset):
    def initialize(self,opt):
        self.opt = opt
        self.root = opt.dataroot
        self.img_size = opt.img_size
        dir_volume = opt.volume_train if opt.train else opt.volume_validation
        self.dir_volume = os.path.join(opt.dataroot, dir_volume)
        self.file_paths = sorted(make_dataset(self.dir_volume, opt.extension))
        self.volume_num = len(self.file_paths)
        self.slices_per_volume = opt.slice_per_volume
        self.degrees_range = opt.degrees_range
        self.scales_range = opt.scales_range
        self.translation_range = opt.translation_range
        self.same_sub = opt.same_sub
        self.device=opt.device
    def __getitem__(self,index):
        filename = self.file_paths[index//self.slices_per_volume]
        # random transforme, record transformation parameters(same 6, different 9)
        img_unresized = nib.load(filename)
        img_data = img_unresized.get_fdata()
        volume = center_crop_or_pad(img_data,self.img_size) # 3D image
        volume = torch.tensor(volume).to(self.device)
        volume = volume.unsqueeze(0)
        degrees = 180*(torch.rand(3)-0.5)
        translation = 40*(torch.rand(3)-0.5)
        scales = (torch.rand(3)+1)/2

        # degrees = np.random.uniform(-90, 90,size=3)
        # translation = np.random.uniform(-20,20, size=3) #np.random.uniform(-50,50, size=3)
        # scales = np.random.uniform(1,1,size=3) if self.same_sub else np.random.uniform(0.5,2,size=3)
        param = torch.cat((degrees, translation), dim=0).squeeze(0) if self.same_sub else torch.cat((degrees, translation, scales), dim=0).squeeze(0)
        matrix = get_affine(param,self.img_size,self.same_sub,self.device).to(self.device)
        central_slice = get_central_slice(matrix.unsqueeze(0),volume,self.device,).to(self.device)#[1,H,W]
        # pading by repeating
        PadV = torch.nn.functional.pad(central_slice.unsqueeze(-1), (self.img_size[-1]-1, 0, 0, 0, 0, 0), mode='replicate', value=0).to(self.device)
        PadV = PadV #[1,H,W,L]
        # output 2D, 3D and parameters
        input_volume = torch.cat((volume,PadV),dim=0).type(torch.cuda.FloatTensor) #[2,H,W,L]
        return (input_volume,param)

    def __len__(self):
        return self.slices_per_volume*self.volume_num

class HumanDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        ###
        ### input moving images
        if opt.pairwise == True:
            dir_A = opt.movingImg_folderName
            self.dir_A = os.path.join(opt.dataroot, dir_A)
            self.A_paths = sorted(make_dataset(self.dir_A, opt.extension))
            # print(len(self.A_paths))
            assert self.A_paths, 'moving images can not find files with extension ' + opt.extension
            ### input fixed images (pairwise or groupwise)
            dir_B = opt.fixedImg_folderName
            self.dir_B = os.path.join(opt.dataroot, dir_B)
            self.B_paths = sorted(make_dataset(self.dir_B, opt.extension))
            assert self.B_paths, 'fixed images can not find files with extension ' + opt.extension
        else:
            self.A_paths = opt.subjectImg
            self.B_paths = opt.templateImg

        self.dataset_size_moving = len(self.A_paths)
        self.dataset_size_fixed = len(self.B_paths)

    def __getitem__(self, index):
        ### moving images
        if self.opt.pairwise == True:
            index_A = np.random.randint(self.dataset_size_moving)
            index_B = np.random.randint(self.dataset_size_fixed)
            if self.A_paths == self.B_paths:
                while index_B == index_A:
                    index_B = np.random.randint(self.dataset_size_fixed)
            tmp_scansA = torch.from_numpy(load_nib_file(self.A_paths[index_A], self.opt))
            tmp_scansB = torch.from_numpy(load_nib_file(self.B_paths[index_B], self.opt))
        else:
            tmp_scansA = torch.from_numpy(load_nib_file(self.A_paths, self.opt))
            tmp_scansB = torch.from_numpy(load_nib_file(self.B_paths, self.opt))


        assert tmp_scansA.shape == tmp_scansB.shape, 'paired scans must have the same shape'

        input_img = [tmp_scansA, tmp_scansB]
        shape = tmp_scansA.shape[:-1]
        zeros = np.zeros((*shape, len(shape)))
        output_img = [tmp_scansB, zeros]
        return (input_img, output_img)

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'Human Dataset'


class MouseDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        ###
        ### input moving images
        if opt.pairwise == True:
            dir_A = opt.movingImg_folderName
            self.dir_A = os.path.join(opt.dataroot, dir_A)
            self.A_paths = sorted(make_dataset(self.dir_A, opt.extension))
            # print(len(self.A_paths))
            assert self.A_paths, 'moving images can not find files with extension ' + opt.extension
            ### input fixed images (pairwise or groupwise)
            dir_B = opt.fixedImg_folderName
            self.dir_B = os.path.join(opt.dataroot, dir_B)
            self.B_paths = sorted(make_dataset(self.dir_B, opt.extension))
            assert self.B_paths, 'fixed images can not find files with extension ' + opt.extension
        else:
            self.A_paths = opt.subjectImg
            self.B_paths = opt.templateImg

        self.dataset_size_moving = len(self.A_paths)
        self.dataset_size_fixed = len(self.B_paths)

    def __getitem__(self, index):
        ### moving images
        if self.opt.pairwise == True:
            index_A = np.random.randint(self.dataset_size_moving)
            index_B = np.random.randint(self.dataset_size_fixed)
            if self.A_paths == self.B_paths:
                while index_B == index_A:
                    index_B = np.random.randint(self.dataset_size_fixed)
            tmp_scansA = torch.from_numpy(load_nib_file(self.A_paths[index_A], self.opt))
            tmp_scansB = torch.from_numpy(load_nib_file(self.B_paths[index_B], self.opt))
        else:
            tmp_scansA = torch.from_numpy(load_nib_file(self.A_paths, self.opt))
            tmp_scansB = torch.from_numpy(load_nib_file(self.B_paths, self.opt))


        assert tmp_scansA.shape == tmp_scansB.shape, 'paired scans must have the same shape'

        input_img = [tmp_scansA, tmp_scansB]
        shape = tmp_scansA.shape[:-1]
        zeros = np.zeros((*shape, len(shape)))
        output_img = [tmp_scansB, zeros]
        return (input_img, output_img)

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'Mouse Dataset'