import pandas as pd
import numpy as np
import torch
import os
import scipy
import torchio as tio
import nibabel as nib
import math
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
import torch
import torchio as tio



# from torchio.transforms.augmentation.spatial import random_affine
# from torchio.transforms import augmentation

def norm_img(img, percentile=100):
    img = (img - np.min(img)) / (np.percentile(img, percentile) - np.min(img))
    return np.clip(img, 0, 1)

# def norm_img(img, percentile=100):
#     img = 2 * (img - np.min(img)) / (np.percentile(img, percentile) - np.min(img)) - 1
#     return np.clip(img, -1, 1)


def is_nifti_file(filename, extension):
    return filename.endswith(extension)


def make_dataset(dir, extension):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_nifti_file(fname, extension):
                path = os.path.join(root, fname)
                images.append(path)

    return images

# def get_bounds(img):
#     #img: torchio.ScalarImage.data
#     #return: idx, a list containing [x_min, x_max, y_min, y_max, z_min, z_max)
#     img = np.squeeze(img.numpy())
#     nz_idx = np.nonzero(img)
#     idx = []
#     for i in nz_idx:
#         idx.append(i.min())
#         idx.append(i.max())

#     return idx

def center_crop_or_pad(input_scan, desired_dimension):
    input_dimension = input_scan.shape
    #print('Input dimension: ', input_dimension, '\ndesired dimension: ', desired_dimension)

    x_lowerbound_target = int(np.floor((desired_dimension[0] - input_dimension[0]) / 2)) if desired_dimension[0] >= input_dimension[0] else 0
    y_lowerbound_target = int(np.floor((desired_dimension[1] - input_dimension[1]) / 2)) if desired_dimension[1] >= input_dimension[1] else 0
    z_lowerbound_target = int(np.floor((desired_dimension[2] - input_dimension[2]) / 2)) if desired_dimension[2] >= input_dimension[2] else 0
    x_upperbound_target = x_lowerbound_target + input_dimension[0] if desired_dimension[0] >= input_dimension[0] else None
    y_upperbound_target = y_lowerbound_target + input_dimension[1] if desired_dimension[1] >= input_dimension[1] else None
    z_upperbound_target = z_lowerbound_target + input_dimension[2] if desired_dimension[2] >= input_dimension[2] else None

    x_lowerbound_input = 0 if desired_dimension[0] >= input_dimension[0] else int(np.floor((input_dimension[0] - desired_dimension[0]) / 2))
    y_lowerbound_input = 0 if desired_dimension[1] >= input_dimension[1] else int(np.floor((input_dimension[1] - desired_dimension[1]) / 2))
    z_lowerbound_input = 0 if desired_dimension[2] >= input_dimension[2] else int(np.floor((input_dimension[2] - desired_dimension[2]) / 2))
    x_upperbound_input = None if desired_dimension[0] >= input_dimension[0] else x_lowerbound_input + desired_dimension[0]
    y_upperbound_input = None if desired_dimension[1] >= input_dimension[1] else y_lowerbound_input + desired_dimension[1]
    z_upperbound_input = None if desired_dimension[2] >= input_dimension[2] else z_lowerbound_input + desired_dimension[2]

    output_scan = np.zeros(desired_dimension).astype(np.double)  

    output_scan[x_lowerbound_target : x_upperbound_target, \
                y_lowerbound_target : y_upperbound_target, \
                z_lowerbound_target : z_upperbound_target] = \
    input_scan[x_lowerbound_input: x_upperbound_input, \
                y_lowerbound_input: y_upperbound_input, \
                z_lowerbound_input: z_upperbound_input]

    return output_scan

def resize(vol, factor, batch_axis=False):
    """
    Resizes an vol by a given factor. This expects the input vol to include a feature dimension.
    Use batch_axis=True to avoid resizing the first (batch) dimension.
    """
    if factor == 1:
        return vol
    else:
        if not batch_axis:
            dim_factors = [factor for _ in vol.shape[:-1]] + [1]
        else:
            dim_factors = [1] + [factor for _ in vol.shape[1:-1]] + [1]
        return scipy.ndimage.interpolation.zoom(vol, dim_factors, order=0)

    
def load_nib_file(filename, opt):
    'implement modifications to the vol - resize, crop, pad, augmentation etc'
    # if opt.specie == 'Mouse':
    #     opt.pad_shape = opt.img_size = (192, 256, 128)
    
    ### Load nib files
    print(filename)
    if filename.endswith('nii') or filename.endswith('nii.gz'):
        import nibabel as nib
        img = nib.load(filename)
        vol = img.get_fdata().squeeze()
        # vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol))
        # vol = norm_img(vol)
        resolution = img.header.get_zooms()
    elif filename.endswith('.h5'):
        import h5py
        with h5py.File(filename, 'r') as f:
            vol = f.get('magnitude')[()][:4, ...]    # (41, 4, 201, 402, 18)
            vol = np.mean(vol, axis=0)
            vol = np.mean(vol, axis=0)
            resolution = f['resolution'][:]
        
    # Read resolution
    scalar0 = resolution[0] / opt.resolution  # for 1st dimension
    scalar1 = resolution[1] / opt.resolution  # for 2nd dimension
    scalar2 = resolution[2] / opt.resolution  # for 3rd dimension

    zoom = 1 
    vol = scipy.ndimage.zoom(vol, (zoom * scalar0, zoom * scalar1, zoom * scalar2), order=3)
    
    ### Data modification
    # Data augmentation - config
    augmentation = opt.Augmentation
    # random_affine = opt.random_affine
    # affine_scales = opt.affine_scales
    # affine_degrees = opt.affine_degrees
    # random_elastic = opt.random_elastic
    # elastic_num_control_points = opt.elastic_num_control_points
    # elastic_max_displacement = opt.elastic_max_displacement
    random_blur = opt.random_blur
    
    # Others - config
    pad_shape = opt.pad_shape
    add_feat_axis = opt.add_feat_axis
    resize_factor = opt.resize_factor
    # add_batch_axis = opt.add_batch_axis
    
    # Data Augmentation
    if augmentation:
        if len(vol.shape) <= 3:
            vol = vol[np.newaxis, ...]
        image = tio.ScalarImage(tensor=vol)
        subject = tio.Subject(t1=image)
        if random_blur:
            add_blur = tio.RandomBlur(
                std=(0, 2), 
                p=0.2
            )
            with_blur = add_blur(subject)
            vol = with_blur.t1.numpy()
        # if random_affine:
        #     # add affine
        #     add_affine = tio.RandomAffine(
        #         scales=affine_scales, 
        #         degrees=affine_degrees, 
        #         )
        #     with_affine = add_affine(subject)
        #     vol = with_affine.t1.numpy()

        # if random_elastic:
        #     # add elastic 
        #     max_displacement = 10, 10, 10  # in x, y and z directions
        #     add_elastic = tio.RandomElasticDeformation(
        #         num_control_points=elastic_num_control_points, 
        #         max_displacement=elastic_max_displacement,
        #         )
        #     with_elastic = add_elastic(subject)
        #     vol = with_elastic.t1.numpy()
        vol = vol.squeeze()

    if pad_shape:
        vol = center_crop_or_pad(vol, pad_shape)

    if add_feat_axis:
        vol = vol[..., np.newaxis]

    if resize_factor != 1:
        vol = resize(vol, resize_factor)

    # if add_batch_axis:
    #     vol = vol[np.newaxis, ...]
    # print(vol.shape)

    return norm_img(vol)

def get_central_slice(matrix,data,device):
    # data should be tensor,[Batch,1,H,W,L]
    batchsize=data.size(0)
    central_slices=[]
    for i in range(batchsize):
      volume=data[i].unsqueeze(0).unsqueeze(0).to(device)
      transform_inp=matrix[i].to(device)
      grid = F.affine_grid(transform_inp, volume.size(), align_corners=True).to(device)
      grid = grid.type_as(volume).to(device)
      y = F.grid_sample(volume, grid, padding_mode='zeros', align_corners=True).to(device)
      slice_index = y.size(-1) // 2
      y_slice = y[...,:,slice_index].squeeze(0).squeeze(0).to(device) #[H,W]
      central_slices.append(y_slice)
    central_slices=torch.stack(central_slices,dim=0)
    return central_slices #[batch,H,W]

def get_affine(params,volume_size,same_sub,device):
    if len(params.size()) > 1:
      batchsize=params.size(0)
    else:
      batchsize=1
    transform_list=[]
    for i in range(batchsize):
      if batchsize > 1:
        param=params[i]
      else:
        param=params.flatten()
      if same_sub:
        scales=torch.tensor([1,1,1]).to(device)
        degrees=param[0:3].to(device)
        translation=param[3:6].to(device)
      else:
        degrees=param[0:3].to(device)
        translation=param[3:6].to(device)
        scales=param[6:9].to(device)



      # Rotation matrix
      # R_x = torch.tensor([[1, 0, 0, 0],
      #                               [0, math.cos(degrees[0]), -math.sin(degrees[0]), 0],
      #                               [0, math.sin(degrees[0]), math.cos(degrees[0]), 0],
      #                               [0, 0, 0, 1],])

      # R_y = torch.tensor([[math.cos(degrees[1]), 0, math.sin(degrees[1]), 0],
      #                               [0, 1, 0, 0],
      #                               [-math.sin(degrees[1]), 0, math.cos(degrees[1]),0],
      #                               [0, 0, 0, 1],])

      # R_z = torch.tensor([[math.cos(degrees[2]), -math.sin(degrees[2]), 0, 0],
      #                               [math.sin(degrees[2]), math.cos(degrees[2]), 0, 0],
      #                               [0, 0, 1, 0],
      #                               [0, 0, 0, 1],])
      # R_x = R_x.to(device) 
      # R_y = R_y.to(device)  
      # R_z = R_z.to(device)                            
      # R = R_x@R_y@R_z

      rotation_matrix = torch.zeros(3, 3, device=device)
      translation_vector = torch.zeros(3, 1, device=device)

      # Extract rigid transformation parameters
      angle_x = (degrees[0]*torch.pi/180).to(device)
      angle_y = (degrees[1]*torch.pi/180).to(device)
      angle_z = (degrees[2]*torch.pi/180).to(device)
      translation_x = (translation[0]/volume_size[0]).to(device)
      translation_y = (translation[1]/volume_size[1]).to(device)
      translation_z = (translation[2]/volume_size[2]).to(device)

      # Compute rotation matrices
      rotation_matrix[0, 0] = torch.cos(angle_y) * torch.cos(angle_z) #输入为角度
      rotation_matrix[0, 1] = torch.cos(angle_z) * torch.sin(angle_x) * torch.sin(angle_y) - torch.cos(angle_x) * torch.sin(angle_z)
      rotation_matrix[0, 2] = torch.cos(angle_x) * torch.cos(angle_z) * torch.sin(angle_y) + torch.sin(angle_x) * torch.sin(angle_z)
      rotation_matrix[1, 0] = torch.cos(angle_y) * torch.sin(angle_z)
      rotation_matrix[1, 1] = torch.cos(angle_x) * torch.cos(angle_z) + torch.sin(angle_x) * torch.sin(angle_y) * torch.sin(angle_z)
      rotation_matrix[1, 2] = -torch.cos(angle_z) * torch.sin(angle_x) + torch.cos(angle_x) * torch.sin(angle_y) * torch.sin(angle_z)
      rotation_matrix[2, 0] = -torch.sin(angle_y)
      rotation_matrix[2, 1] = torch.cos(angle_y) * torch.sin(angle_x)
      rotation_matrix[2, 2] = torch.cos(angle_x) * torch.cos(angle_y)

      rotation_matrix = rotation_matrix.to(device)

      # Compute translation vectors
      translation_vector[0, 0] = translation_x
      translation_vector[1, 0] = translation_y
      translation_vector[2, 0] = translation_z
      translation_vector = translation_vector.to(device)
      # Concatenate rotation matrices and translation vectors to form affine transformation matrices
      rigid_transformation_matrix = torch.cat((rotation_matrix, translation_vector), dim=1).to(device)
      if same_sub:
        affine_transformation_matrix = rigid_transformation_matrix.to(device)
      else:
        scales_x = 1/scales[0]
        scales_y = 1/scales[1]
        scales_z = 1/scales[2]
        scale_matrix = torch.zeros(3, 4, device=device)
        scale_matrix[0,0] = scales_x
        scale_matrix[1,1] = scales_y
        scale_matrix[2,2] = scales_z
        bottom = torch.tensor(0,0,0,1)
        affine = torch.cat((rigid_transformation_matrix,bottom),dim=0) @ torch.cat((scale_matrix,bottom),dim=0)
        affine_transformation_matrix = affine[0:3.:]
      # Translation matrix
      # T = torch.tensor([[1, 0, 0, -translation[0]/(volume_size[0]/2)],
      #                             [0, 1, 0, -translation[1]/(volume_size[1]/2)],
      #                             [0, 0, 1, -translation[2]/(volume_size[2]/2)],
      #                             [0, 0, 0, 1],])
      # T=T.float().to(device)





      # # Scaling matrix
      # S = torch.tensor([[1/scales[0], 0, 0, 0],
      #                             [0, 1/scales[1], 0, 0],
      #                             [0, 0, 1/scales[2], 0],
      #                             [0, 0, 0, 1],])
      # S = S.float().to(device)
      # transform=R@T@S
      # transform=transform.to(device)
      # transform_list.append(transform[0:3])
      transform_list.append(affine_transformation_matrix)
    transform = torch.stack(transform_list)
    return transform


