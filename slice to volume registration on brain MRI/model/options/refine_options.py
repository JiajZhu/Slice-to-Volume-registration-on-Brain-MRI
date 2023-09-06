import ml_collections
import torch.nn as nn
from timm.models.layers import to_3tuple
'''
********************************************************
                   Swin Transformer
********************************************************
if_transskip (bool): Enable skip connections from Transformer Blocks
if_convskip (bool): Enable skip connections from Convolutional Blocks
patch_size (int | tuple(int)): Patch size. Default: 4
in_chans (int): Number of input image channels. Default: 2 (for moving and fixed images)
embed_dim (int): Patch embedding dimension. Default: 96
depths (tuple(int)): Depth of each Swin Transformer layer.
num_heads (tuple(int)): Number of attention heads in different layers.
window_size (tuple(int)): Image size should be divisible by window size, 
                     e.g., if image has a size of (160, 192, 224), then the window size can be (5, 6, 7)
mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
pat_merg_rf (int): Embed_dim reduction factor in patch merging, e.g., N*C->N/4*C if set to four. Default: 4. 
qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
drop_rate (float): Dropout rate. Default: 0
drop_path_rate (float): Stochastic depth rate. Default: 0.1
ape (bool): Enable learnable position embedding. Default: False
spe (bool): Enable sinusoidal position embedding. Default: False
rpe (bool): Enable relative position embedding. Default: True
patch_norm (bool): If True, add normalization after patch embedding. Default: True
use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False 
                       (Carried over from Swin Transformer, it is not needed)
out_indices (tuple(int)): Indices of Transformer blocks to output features. Default: (0, 1, 2, 3)
reg_head_chan (int): Number of channels in the registration head (i.e., the final convolutional layer) 
img_size (int | tuple(int)): Input image size, e.g., (160, 192, 224)
'''
def get_CycleTransMorph_config(type='train', specie='Human'):
    '''
    Trainable params: 15,201,579
    '''
    config = ml_collections.ConfigDict()
        
    #####################################################
    # Model configurations
    #####################################################
    config = ml_collections.ConfigDict()
    config.patch_size = 4
    config.in_chans = 2
    config.reg_head_chan = 16
    
    config.depths = [2, 2, 6, 2]
    config.depths_decoder = [2, 2, 6, 2]
    config.num_heads=[3, 6, 12, 24]
    config.mlp_ratio = 4.
    config.qkv_bias = True
    config.qk_scale = None
    config.drop_rate = 0.
    config.attn_drop_rate = 0.
    config.drop_path_rate = 0.1
    config.norm_layer = nn.LayerNorm
    config.ape = False
    config.patch_norm = True

    config.int_steps= 7
    config.int_downsize = 2
    config.final_upsample = 'expand_first'

    config.device = 'cuda:0'

    #####################################################
    # Data loader configurations
    #####################################################
    config.specie = specie
    #####################################################
    # Human
    #####################################################
    if config.specie == 'Human':
        config.img_size = (160, 192, 192)
        config.window_size = (5, 6, 6)
        config.embed_dim = 48
        config.dataroot = '/content/drive/MyDrive/Guo_Lab/Registration/Data/unimodal_dataset/human/Aging_Dataset_XYF_BrainAgePaper'
        config.resolution = 1.0
        config.train_iteration = 100
        # config.validation_iteration = 10
        config.checkpoints_dir = 'saved_model_CycleIC_Human'
        config.use_checkpoint = False
        config.checkpoint = 'Best_model_MS-SSIM.pt'
        config.pad_shape = config.img_size
        config.extension = '.nii.gz'


        config.add_feat_axis = True
        config.resize_factor = 1.
        # config.add_batch_axis = False
        
        config.Augmentation = False
        config.random_affine = True
        config.affine_scales = 0.5
        config.affine_degrees = 10
        config.random_elastic = True
        config.elastic_num_control_points = 5
        config.elastic_max_displacement = 5
        
        #####################################################
        # Train configurations
        #####################################################
        config.nThreads = 0
        config.gpu_ids = 0
        config.max_epoch = 1500
        config.similarity_loss_type = 'MS-SSIM' # NCC, MSE, SSIM, MI, MS-SSIM
        config.smoothness_loss = 'l2'
        config.smoothness_weight = 0.5
        config.NJ_weight = 10
        config.IC_weight_image = 1
        config.IC_weight_flow = 0.05
        config.lr = 5e-4

    #####################################################
    # Mouse
    #####################################################
    elif config.specie == 'Mouse':
        config.img_size = (160, 224, 96)
        config.window_size = (5, 7, 3)
        config.embed_dim = 48
        config.dataroot = '/content/drive/MyDrive/Guo_Lab/Registration/Data/unimodal_dataset/Mouse/Affine_space/Mouse_Mean'
        config.resolution = 0.08
        config.train_iteration = 25
        config.validation_iteration = 5 
        config.checkpoints_dir = 'saved_model_CycleIC_Mouse'
        config.use_checkpoint = True
        config.checkpoint = 'Best_model_SSIM_v2.pt'
        config.pad_shape = (192, 256, 128)

        config.extension = '.nii.gz'

        config.add_feat_axis = True
        config.resize_factor = 1.
        # config.add_batch_axis = False
        
        config.Augmentation = False
        config.random_affine = True
        config.affine_scales = 0.5
        config.affine_degrees = 10
        config.random_elastic = True
        config.elastic_num_control_points = 5
        config.elastic_max_displacement = 5
        
        #####################################################
        # Train configurations
        #####################################################
        config.nThreads = 0
        config.gpu_ids = 0
        config.max_epoch = 1500
        config.similarity_loss_type = 'SSIM' # NCC, MSE, SSIM, MI, MS-SSIM
        config.smoothness_loss = 'l2'
        config.smoothness_weight = 0.5
        config.NJ_weight = 10
        config.IC_weight_image = 1
        config.IC_weight_flow = 0.5
        config.lr = 5e-4

    config.pairwise = True
    if type == 'train':
        config.movingImg_folderName = 'subset_wo_oasis/train'
        config.fixedImg_folderName = 'subset_wo_oasis/train'
        config.batchsize = 1
    elif type == 'validation':
        config.movingImg_folderName = 'subset_wo_oasis/validation'
        config.fixedImg_folderName = 'subset_wo_oasis/validation'
        config.batchsize = 1
    elif type == 'test':
        config.movingImg_folderName = 'subset_wo_oasis/test'
        config.fixedImg_folderName = 'subset_wo_oasis/test'
        config.batchsize = 1
    elif type == 'template':
        config.movingImg_folderName = None
        config.fixedImg_folderName = None
        config.batchsize = 1
        
    return config