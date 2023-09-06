import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from torch.distributions.normal import Normal
import torch.nn.functional as nnf
# import models.configs_TransMorph as configs
from .modelio import LoadableModel, store_config_args
from . import SpatialTransformer



class convBlock(nn.Module):
    def __init__(self,inplace,outplace,kernel_size=3,padding=1):
        super().__init__()
        
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(inplace,outplace,kernel_size=kernel_size,padding=padding,bias=False)
        self.bn1 = nn.BatchNorm3d(outplace)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class VGG83D(nn.Module):
    def __init__(self,inplace):
        super().__init__()
        
        ly = [64,128,256,512]
        
        self.ly = ly
        
        self.maxp = nn.MaxPool3d(2)
        
        self.conv11 = convBlock(inplace,ly[0])
        self.conv12 = convBlock(ly[0],ly[0])
        
        self.conv21 = convBlock(ly[0],ly[1])
        self.conv22 = convBlock(ly[1],ly[1])
        
        self.conv31 = convBlock(ly[1],ly[2])
        self.conv32 = convBlock(ly[2],ly[2])
        
        self.conv41 = convBlock(ly[2],ly[3])
        self.conv42 = convBlock(ly[3],ly[3])
        
    def forward(self,x):

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.maxp(x)
 
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.maxp(x)

        x = self.conv31(x)
        x = self.conv32(x)
        x = self.maxp(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = self.maxp(x)

        return x

class VGG113D(nn.Module):
    def __init__(self,inplace):
        super().__init__()
        
        #ly = [16,32,64,64,128,128,128,128]
        ly = [16,32,64,64,128,128,128,128]
        
        self.ly = ly
        
        self.maxp1 = nn.MaxPool3d(2)
        self.maxp2 = nn.MaxPool3d(2)
        self.maxp3 = nn.MaxPool3d(2)
        self.maxp4 = nn.MaxPool3d(2)
        self.maxp5 = nn.MaxPool3d(2)
        
        self.conv11 = convBlock(inplace,ly[0])

        self.conv21 = convBlock(ly[0],ly[1])
        
        self.conv31 = convBlock(ly[1],ly[2])
        self.conv32 = convBlock(ly[2],ly[3])
        
        self.conv41 = convBlock(ly[3],ly[4])
        self.conv42 = convBlock(ly[4],ly[5])
        
        self.conv51 = convBlock(ly[5],ly[6])
        self.conv52 = convBlock(ly[6],ly[7])
        
    def forward(self,x):

        x = self.conv11(x)
        x = self.maxp1(x)

        x = self.conv21(x)
        x = self.maxp2(x)
 
        x = self.conv31(x)
        x = self.conv32(x)
        x = self.maxp3(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = self.maxp4(x)

        x = self.conv51(x)
        x = self.conv52(x)
        x = self.maxp5(x)

        return x

class VGG16(nn.Module):
    def __init__(self,inplace):
        super().__init__()
        
        ly = [64,128,256,512,512]
        
        self.ly = ly
        
        self.maxp = nn.MaxPool2d(2)
        
        self.conv11 = convBlock(inplace,ly[0])
        self.conv12 = convBlock(ly[0],ly[0])
        
        self.conv21 = convBlock(ly[0],ly[1])
        self.conv22 = convBlock(ly[1],ly[1])
        
        self.conv31 = convBlock(ly[1],ly[2])
        self.conv32 = convBlock(ly[2],ly[2])
        self.conv33 = convBlock(ly[2],ly[2])
        
        self.conv41 = convBlock(ly[2],ly[3])
        self.conv42 = convBlock(ly[3],ly[3])
        self.conv43 = convBlock(ly[3],ly[3])
        
        self.conv51 = convBlock(ly[3],ly[3])
        self.conv52 = convBlock(ly[3],ly[3])
        self.conv53 = convBlock(ly[3],ly[3])
        
    def forward(self,x):

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.maxp(x)
 
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.maxp(x)

        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)
        x = self.maxp(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = self.conv43(x)
        x = self.maxp(x)
        
        x = self.conv51(x)
        x = self.conv52(x)
        x = self.conv53(x)

        return x    

class linear_predict(nn.Module):
    def __init__(self, in_channels=128,pool_layers=5,img_size=(192,192,192),same_sub=False, hidden_features=4096,act_layer=nn.GELU, drop=0.):
        super().__init__()
        # out_features = 12
        out_features = 6 if same_sub else 9
        img_size_pre = list(i*(2**(-pool_layers)) for i in img_size)
        in_features = np.prod(img_size_pre) * in_channels
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc1=nn.Linear(int(in_features),hidden_features)
        self.fc2=nn.Linear(hidden_features,out_features)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x = self.fc(x)


        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        # x = self.act(x)
        # x = self.drop(x)
        # x = self.fc3(x)
        # x = self.final(x)
        
        return x


class VGG_registration(LoadableModel):
  @store_config_args
  def __init__(self,config):
    super().__init__()
    self.VGG113D = VGG113D(inplace=2)
    self.norm = nn.BatchNorm3d(128)
    self.predict = linear_predict(img_size=config.img_size,same_sub=config.same_sub)
  def forward(self,x):
    x = self.VGG113D(x)
    x = self.norm(x)
    x = torch.flatten(x,start_dim=1)
    x = self.predict(x)
    # x = x.view(-1, 3, 4)


    return x