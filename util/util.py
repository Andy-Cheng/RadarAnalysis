import numpy as np
from scipy.io import loadmat
import os
import open3d as o3d
import torch
import torch.nn.functional as F

rad2deg = 180./np.pi
deg2rad = np.pi/180.

def get_rdr_lidar_frame_difference(data_root, target_seqs):
    rdr_lidar_frame_difference = {}
    for seq in target_seqs:
        with open(os.path.join(data_root, str(seq), 'info_calib', 'calib_radar_lidar.txt'), 'r') as f:
            line = f.readlines()[1].strip()
        rdr_lidar_frame_difference[str(seq)] = int(line.split(',')[0])
    return rdr_lidar_frame_difference

def load_physical_values():
    temp_values = loadmat(os.path.join('/mnt/ssd1/kradar_dataset', 'resources', 'info_arr.mat'))
    arr_range = temp_values['arrRange']
    arr_azimuth = temp_values['arrAzimuth']*deg2rad
    arr_elevation = temp_values['arrElevation']*deg2rad
    arr_range = arr_range.flatten()
    arr_azimuth = arr_azimuth.flatten()
    arr_elevation = arr_elevation.flatten()
    arr_doppler = loadmat(os.path.join('/mnt/ssd1/kradar_dataset', 'resources', 'arr_doppler.mat'))['arr_doppler']
    arr_doppler = arr_doppler.flatten()
    return arr_range, arr_azimuth, arr_elevation, arr_doppler

def get_physical_values_xyz(xyz_min, xyz_max, grid_size):
    x_min, y_min, z_min = xyz_min
    x_max, y_max, z_max = xyz_max
    arr_z_cb = np.arange(z_min, z_max, grid_size)
    arr_y_cb = np.arange(y_min, y_max, grid_size)
    arr_x_cb = np.arange(x_min, x_max, grid_size)
    grid_size = 0.4
    return arr_x_cb, arr_y_cb, arr_z_cb

def polar_to_cart(rae_coord):
    r, a, e = rae_coord.T
    x = r * np.cos(a) * np.cos(e)
    y = r * np.sin(a) * np.cos(e)
    z = r * np.sin(e)
    return np.stack((x, y, z), axis=1)

def pick_top(radar_tensor, arr_x_axis, arr_y_axis, arr_z_axis, pick_rate, p2c):
    quantile_rate = 1.0 - pick_rate
    z_ind, y_ind, x_ind = np.where(radar_tensor > np.quantile(radar_tensor, quantile_rate))
    z_coord = arr_z_axis[z_ind][:, None]
    y_coord = arr_y_axis[y_ind][:, None]
    x_coord = arr_x_axis[x_ind][:, None]
    xyz_coords = np.concatenate((x_coord, y_coord, z_coord), axis=1)
    if p2c:
        xyz_coords = polar_to_cart(xyz_coords)
    return xyz_coords

def apply_cfar(rdr_tensor, num_guard_cells, num_training_cells, threshold_factor1, threshold_factor2):
    # Get rdr_tensor dimensions
    elevation_dim, azimuth_dim, range_dim = rdr_tensor.shape
    # r: 256, e: 37, a: 107
    # Initialize an empty rdr_tensor to store CFAR output
    cfar_output = np.zeros_like(rdr_tensor)

    # Loop through each cell in the 4D rdr_tensor
    for r in range(range_dim):
        for e in range(elevation_dim):
            for a in range(azimuth_dim):
                # Define the grid for training cells
                training_cells = []
                for tr in range(r-num_guard_cells-num_training_cells, min(r+num_guard_cells+num_training_cells+1, range_dim)):
                    for te in range(e-num_guard_cells-num_training_cells, min(e+num_guard_cells+num_training_cells+1, elevation_dim)):
                        for ta in range(a-num_guard_cells-num_training_cells, min(a+num_guard_cells+num_training_cells+1, azimuth_dim)):
                            if (abs(tr-r)>num_guard_cells and
                                abs(te-e)>num_guard_cells and abs(ta-a)>num_guard_cells):
                                if 0 <= tr < range_dim and 0 <= te < elevation_dim and 0 <= ta < azimuth_dim:
                                    training_cells.append(rdr_tensor[te, ta, tr])
                # Calculate noise threshold based on training cells
                threshold = np.mean(training_cells) * threshold_factor1 + np.std(training_cells) * threshold_factor2
                # Test against the threshold
                if rdr_tensor[te, ta, tr] > threshold:
                    cfar_output[e, a, r] = 1
    return cfar_output



def create_cfar_kernel(num_guard_cells, num_training_cells, cuda=True):
    # Define the shape of the convolution kernel
    kernel_shape = (
        2*num_training_cells + 2*num_guard_cells + 1, 
        2*num_training_cells + 2*num_guard_cells + 1, 
        2*num_training_cells + 2*num_guard_cells + 1
    )
    # Create a kernel filled with ones
    kernel = torch.ones(kernel_shape, dtype=torch.float32)
    
    # Zero out the guard cells and normalize kernel
    kernel[
        num_training_cells:num_training_cells+2*num_guard_cells+1, 
        num_training_cells:num_training_cells+2*num_guard_cells+1,
        num_training_cells:num_training_cells+2*num_guard_cells+1
    ] = 0
    kernel = kernel.cuda()
    num_effective_cells = torch.sum(kernel)
    kernel = kernel / num_effective_cells  # Normalizing the kernel
    return kernel

def apply_cfar_torch(rdr_tensor, kernel, threshold_factor1, threshold_factor2, arr_x_axis, arr_y_axis, arr_z_axis, pad_length, p2c, add_padding=True):
    with torch.no_grad():
        # Convert numpy array to PyTorch tensor
        rdr_tensor = torch.tensor(rdr_tensor, dtype=torch.float32).to(kernel.device)       
        # Add additional dimensions to rdr_tensor and kernel for batch and channel
        rdr_tensor = rdr_tensor.unsqueeze(0).unsqueeze(0)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        padding = kernel.shape[0]//2 if add_padding else 0
        # Calculate the average and standard deviation of the training cells
        avg = F.conv3d(rdr_tensor, kernel, padding=padding)
        std = torch.sqrt(F.conv3d(rdr_tensor**2, kernel, padding=padding) - avg**2)
        # Calculate threshold
        threshold = avg * threshold_factor1 + std * threshold_factor2
        rdr_tensor_cropped = rdr_tensor[:, :, pad_length:-pad_length, pad_length:-pad_length, pad_length:-pad_length]
        # Create CFAR output tensor
        z_ind, y_ind, x_ind = (rdr_tensor_cropped[0, 0] > threshold[0, 0]).cpu().numpy().nonzero()
        z_coord = arr_z_axis[z_ind][:, None]
        y_coord = arr_y_axis[y_ind][:, None]
        x_coord = arr_x_axis[x_ind][:, None]
        xyz_coords = np.concatenate((x_coord, y_coord, z_coord), axis=1)
        if p2c:
            xyz_coords = polar_to_cart(xyz_coords)
        return xyz_coords
    
def closest_odd_number(float_num):
    rounded_num = round(float_num)
    closest_odd = rounded_num + 1 if rounded_num % 2 == 0 else rounded_num
    return closest_odd