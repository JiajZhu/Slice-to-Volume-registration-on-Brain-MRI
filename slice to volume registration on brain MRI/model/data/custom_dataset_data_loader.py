### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
### This script is modified based on the pix2pixHD official implementation (see license above)
### https://github.com/NVIDIA/pix2pixHD
import torch.utils.data
from .base_data_loader import BaseDataLoader


def CreateDataset(opt):
    dataset = None
    from .dataset import S2VDataset
    dataset = S2VDataset()
    # print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self, opt):
        if opt.specie == 'Human':
            return 'Paired/human dataloader'
        elif opt.specie == 'Mouse':
            return 'Paired/mouse dataloader'
    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchsize,
            shuffle=True,
            num_workers=int(opt.nThreads), pin_memory=False)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)