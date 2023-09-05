import numpy as np
from scipy.io import loadmat
import os
import open3d as o3d

def load_physical_values():
    temp_values = loadmat(os.path.join('/mnt/ssd1/kradar_dataset', 'resources', 'info_arr.mat'))
    arr_range = temp_values['arrRange']
    deg2rad = np.pi/180.
    arr_azimuth = temp_values['arrAzimuth']*deg2rad
    arr_elevation = temp_values['arrElevation']*deg2rad
    arr_range = arr_range.flatten()
    arr_azimuth = arr_azimuth.flatten()
    arr_elevation = arr_elevation.flatten()
    arr_doppler = loadmat(os.path.join('/mnt/ssd1/kradar_dataset', 'resources', 'arr_doppler.mat'))['arr_doppler']
    arr_doppler = arr_doppler.flatten()
    return arr_range, arr_azimuth, arr_elevation, arr_doppler


def apply_cfar(matrix, num_guard_cells, num_training_cells, threshold_factor1, threshold_factor2):
    # Get matrix dimensions
    elevation_dim, azimuth_dim, range_dim = matrix.shape
    # r: 256, e: 37, a: 107
    # Initialize an empty matrix to store CFAR output
    cfar_output = np.zeros_like(matrix)

    # Loop through each cell in the 4D matrix
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
                                    training_cells.append(matrix[te, ta, tr])
                # Calculate noise threshold based on training cells
                threshold = np.mean(training_cells) * threshold_factor1 + np.std(training_cells) * threshold_factor2
                # Test against the threshold
                if matrix[te, ta, tr] > threshold:
                    cfar_output[e, a, r] = 1
    return cfar_output

def polar_to_cart(ear_coord):
    e, a, r = ear_coord.T
    x = r * np.cos(a) * np.cos(e)
    y = r * np.sin(a) * np.cos(e)
    z = r * np.sin(e)
    return np.stack((x, y, z), axis=1)



# arr_range, arr_azimuth, arr_elevation, arr_doppler = load_physical_values()
# radar_ear = np.load('/mnt/ssd1/kradar_dataset/radar_tensor/11/radar_DEAR_D_downsampled_2/tesseract_00034.npy').max(axis=0)

# # Apply 4D CFAR detection
# num_guard_cells = 2
# num_training_cells = 10
# threshold_factor1 = 10
# threshold_factor2 = 5


# cfar_output = apply_cfar(radar_ear, num_guard_cells, num_training_cells, threshold_factor1, threshold_factor2)
# # cfar_output will contain ones at positions where peaks are detected and zeros elsewhere

# e,a,r = np.where(cfar_output == 1)

# # coord_ears = np.column_stack((arr_elevation[e], arr_azimuth[a], arr_range[r]))
# # xyz_coords = polar_to_cart(coord_ears)

# coord_ear = (arr_elevation[e[0]], arr_azimuth[a[0]], arr_range[r[0]])
# x,y,z = polar_to_cart(coord_ear)
# xyz_coords = np.array([[x,y,z]])
# for i in range(1, len(e)):

#     coord_ear = (arr_elevation[e[i]], arr_azimuth[a[i]], arr_range[r[i]])
#     x, y, z = polar_to_cart(coord_ear)
#     xyz_coords = np.append(xyz_coords, [[x,y,z]], axis=0) 

# # print(xyz_coords.shape)
# print(xyz_coords)


# save_path = '/mnt/nas_kradar/kradar_dataset/dir_all/11/radar/ca_cfar/00001.pcd'
# ## write .pcd file
# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(xyz_coords)
# o3d.io.write_point_cloud(save_path, point_cloud)


# ## read .pcd file
# pcd = o3d.io.read_point_cloud(save_path)
# out_arr = np.asarray(pcd.points)  
# print("output array from input list : ", out_arr)  


# # TODO: pick top N % of points on radar zyx tensor
pick_rate = 0.005 # pick top 0.1% of the points
quantile_rate = 1.0 - pick_rate


arr_z_cb = np.arange(-30, 30, 0.4)
arr_y_cb = np.arange(-80, 80, 0.4)
arr_x_cb = np.arange(0, 100, 0.4)
grid_size = 0.4
z_min, z_max = arr_z_cb[0], arr_z_cb[-1]
y_min, y_max = arr_y_cb[0], arr_y_cb[-1]
x_min, x_max = arr_x_cb[0], arr_x_cb[-1]

# rdr_tensor_zyx_path = '/mnt/ssd1/kradar_dataset/radar_tensor_zyx/11/cube_00034.npy'
# rdr_tensor_zyx = np.flip(np.load(rdr_tensor_zyx_path), axis=0)
# z_ind, y_ind, x_ind = np.where(rdr_tensor_zyx > np.quantile(rdr_tensor_zyx, quantile_rate))
# z_pc_coord = ((z_min + z_ind * grid_size) - grid_size / 2)[:, None]
# y_pc_coord = ((y_min + y_ind * grid_size) - grid_size / 2)[:, None]
# x_pc_coord = ((x_min + x_ind * grid_size) - grid_size / 2)[:, None]
# print("-- cart: ind --")
# print(z_ind, y_ind, x_ind)
# print("-- cart pc coord --")
# print(z_pc_coord, y_pc_coord, x_pc_coord)

# xyz_coords = np.concatenate((x_pc_coord, y_pc_coord, z_pc_coord), axis=1)
# print(xyz_coords)
# save_path = '/mnt/nas_kradar/kradar_dataset/dir_all/11/radar/top_10/00001.pcd'
# ## write .pcd file
# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(xyz_coords)
# o3d.io.write_point_cloud(save_path, point_cloud)




### polar top 0.1%
arr_range, arr_azimuth, arr_elevation, arr_doppler = load_physical_values()
radar_ear = np.load('/mnt/ssd1/kradar_dataset/radar_tensor/11/radar_DEAR_D_downsampled_2/tesseract_00034.npy').max(axis=0)

rad2deg = 180./np.pi
e_ind, a_ind, r_ind = np.where(radar_ear > np.quantile(radar_ear, quantile_rate))
e_coord = arr_elevation[e_ind][:, None]
a_coord = arr_azimuth[a_ind][:, None]
r_coord = arr_range[r_ind][:, None]
# print("-- polar pc coord -- ")
# print(r_pc_coord, e_pc_coord, a_pc_coord)
ear_coords = np.concatenate((e_coord, a_coord, r_coord), axis=1)
pc_coords = polar_to_cart(ear_coords)
save_path = '/mnt/nas_kradar/kradar_dataset/dir_all/11/radar/polar_top_10/00001.pcd'
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(pc_coords)
o3d.io.write_point_cloud(save_path, point_cloud)

# TODO: Nx3 array of points under polar coordinates -> Nx3 cartesian coordinates


# pcd_path = '/mnt/nas_kradar/kradar_dataset/dir_all/11/radar/polar_top_10/00001.pcd'
# point_cloud = np.asarray(o3d.io.read_point_cloud(pcd_path).points)

# print(point_cloud.shape)

