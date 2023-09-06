import ml_collections
import torch.nn as nn
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
    if config.specie == 'Human':
        config.img_size = (160, 192, 192)
        config.window_size = (5, 6, 6)
        config.embed_dim = 48
        config.dataroot = '/media/sail/HDD18T/BME_Grad_Project/Chenghao_CycleICTD/Dataset/Human/Human_large/'
        config.resolution = 1.0
        config.train_iteration = 1000
        config.validation_iteration = 1
        config.checkpoints_dir = '/saved_model_CycleIC_Human'
        config.use_checkpoint = True
        config.checkpoint = '/Best_model_SSIM.pt'
        config.pad_shape = config.img_size
        
        config.add_feat_axis = True
        config.resize_factor = 1.
        # config.add_batch_axis = False
        
        config.Augmentation = False
        # config.random_affine = True
        # config.affine_scales = 0.5
        # config.affine_degrees = 10
        # config.random_elastic = True
        # config.elastic_num_control_points = 5
        # config.elastic_max_displacement = 5
        config.random_blur = True
        #####################################################
        # Train configurations
        #####################################################
        config.nThreads = 0
        config.gpu_ids = 0
        config.max_epoch = 1500
        config.similarity_loss_type = 'SSIM' # NCC, MSE, SSIM, MI, MS-SSIM
        config.smoothness_loss = 'l2'
        config.smoothness_weight = 0.5
        config.NJ_weight = 1000
        config.IC_weight_image = 10
        config.IC_weight_flow = 1
        config.lr = 5e-4

    elif config.specie == 'Mouse':
        config.img_size = (160, 224, 96)
        config.window_size = (5, 7, 3)
        config.embed_dim = 48
        config.dataroot = '/media/sail/HDD18T/BME_Grad_Project/Chenghao_CycleICTD/Dataset/Mouse/Mouse_Mean'
        config.resolution = 0.08
        config.train_iteration = 25
        config.validation_iteration = 5 
        config.checkpoints_dir = 'saved_model_CycleIC_Mouse'
        config.use_checkpoint = False
        config.checkpoint = 'Best_model_MS-SSIM.pt'
        config.pad_shape = (192, 256, 128)
        
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
        config.IC_weight_flow = 0.1
        config.lr = 5e-4

    config.type = type
    if type == 'train':
        config.movingImg_folerName = 'train'
        config.fixedImg_folerName = 'train'
        config.batchsize = 1
    elif type == 'validation':
        config.movingImg_folerName = 'validation'
        config.fixedImg_folerName = 'validation'
        config.batchsize = 1
    elif type == 'test':
        config.movingImg_folerName = 'test'
        config.fixedImg_folerName = 'test'
        config.batchsize = 1
    config.extension = '.nii.gz'

    config.pairwise = False
    if config.pairwise == True:
        config.movingImg_folerName = 'subset1'
        config.fixedImg_folerName = 'age_specific_template_10_age_points/age_specific_template_10_age_points_iteration_0'
    else:
        config.subjectImg = None
        config.templateImg = None
    config.batchsize = 1


    
    return config