from util.util import *
from pathlib import Path
from tqdm import tqdm
import json

xyz_min, xyz_max = [0, -80, -30], [100, 80, 30]
grid_size = 0.4
arr_x_cb, arr_y_cb, arr_z_cb = get_physical_values_xyz(xyz_min, xyz_max, grid_size)
arr_range, arr_azimuth, arr_elevation, arr_doppler = load_physical_values()
kradar_root = '/mnt/nas_kradar/kradar_dataset/dir_all'

def pick_rate(rdr_path, kradar_root, rdr_frame_offset, p2c,  arr_x_cb, arr_y_cb, arr_z_cb, pick_range=[0.05, 0.21], step=0.01):
    path_parts = Path(rdr_path).parts
    seq = path_parts[5]
    frame = '{:05}'.format(int(path_parts[-1].split('.')[0].split('_')[1]) - rdr_frame_offset)
    if int(frame) < 0:
        return
    rdr_tensor = np.load(rdr_path)
    if len(rdr_tensor.shape) == 4:
        rdr_tensor = rdr_tensor.max(axis=0)
    rdr_tensor = np.flip(rdr_tensor, axis=0).copy() 
    for pick_rate in np.arange(*pick_range, step):
        coord_type = 'polar' if p2c else 'cart'
        save_path = os.path.join(kradar_root, seq, 'radar', f'{coord_type}_top_{int(pick_rate*100)}')
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, f'{frame}.pcd')
        xyz_coords = pick_top(rdr_tensor, arr_x_cb, arr_y_cb, arr_z_cb, pick_rate, p2c)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(xyz_coords)
        o3d.io.write_point_cloud(save_path, point_cloud)     

def cfar(rdr_path, kradar_root, rdr_frame_offset, p2c,  arr_x_cb, arr_y_cb, arr_z_cb, \
         train_cell_sizes=[15, 25], guard_cell_sizes=[5, 10], \
            threshold_factors1=[1, 1.5], threshold_factors2=[0.75, 1.5]):
    path_parts = Path(rdr_path).parts
    seq = path_parts[5]
    frame = '{:05}'.format(int(path_parts[-1].split('.')[0].split('_')[1]) - rdr_frame_offset)
    if int(frame) < 0:
        return
    rdr_tensor = np.load(rdr_path)
    rdr_tensor[rdr_tensor<0]  = 0.
    if len(rdr_tensor.shape) == 4:
        rdr_tensor = rdr_tensor.max(axis=0)
    rdr_tensor = np.flip(rdr_tensor, axis=0).copy() 
    for train_cell_size in train_cell_sizes:
        for guard_cell_size in guard_cell_sizes:
            if train_cell_size < guard_cell_size:
                continue
            for threshold_factor1 in threshold_factors1:
                for threshold_factor2 in threshold_factors2:
                    # print(f'{train_cell_size}, {guard_cell_size}, {threshold_factor1}, {threshold_factor2}')
                    coord_type = 'polar' if p2c else 'cart'
                    save_path = os.path.join(kradar_root, seq, 'radar', f'{coord_type}_cacfar_{train_cell_size}_{guard_cell_size}_{int(threshold_factor1*100)}_{int(threshold_factor2*100)}')
                    os.makedirs(save_path, exist_ok=True)
                    save_path = os.path.join(save_path, f'{frame}.pcd')
                    xyz_coords = apply_cfar_torch(rdr_tensor, guard_cell_size, train_cell_size, \
                                                  threshold_factor1, threshold_factor2, \
                                                    arr_x_cb, arr_y_cb, arr_z_cb, p2c)
                    point_cloud = o3d.geometry.PointCloud()
                    point_cloud.points = o3d.utility.Vector3dVector(xyz_coords)
                    o3d.io.write_point_cloud(save_path, point_cloud)
                    
def object_cfar(rdr_path, kradar_root, rdr_frame_offset, p2c,  arr_x_cb, arr_y_cb, arr_z_cb, \
         train_cell_sizes=[15], guard_cell_sizes=[5], \
            threshold_factors1=[1, 1.5], threshold_factors2=[1.5]):
    grid_size = arr_x_cb[1] - arr_x_cb[0]
    path_parts = Path(rdr_path).parts
    seq = path_parts[5]
    frame = '{:05}'.format(int(path_parts[-1].split('.')[0].split('_')[1]) - rdr_frame_offset)
    if int(frame) < 0:
        return False
    label_path = os.path.join(kradar_root, seq, 'label', f'{frame}.json')
    objs = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            objs = json.load(f)
    rdr_tensor = np.load(rdr_path)
    rdr_tensor[rdr_tensor<0]  = 0.
    if len(rdr_tensor.shape) == 4:
        rdr_tensor = rdr_tensor.max(axis=0)
    rdr_tensor = np.flip(rdr_tensor, axis=0).copy() 
    for train_cell_size in train_cell_sizes:
        for guard_cell_size in guard_cell_sizes:
            if train_cell_size < guard_cell_size:
                continue
            # pad edge with maximum value to prevent selecting edge cells to be CFAR points
            pad_length = train_cell_size + guard_cell_size
            padded_rdr_tensor = np.pad(rdr_tensor, ((pad_length, pad_length), (pad_length, pad_length), (pad_length, pad_length)), 'maximum')
            # padded_arr_x_cb = np.pad(arr_x_cb, (pad_length, pad_length), 'edge')
            # padded_arr_y_cb = np.pad(arr_y_cb, (pad_length, pad_length), 'edge')
            # padded_arr_z_cb = np.pad(arr_z_cb, (pad_length, pad_length), 'edge')
            kernel_3D = create_cfar_kernel(train_cell_size, guard_cell_size, cuda=True)
            for threshold_factor1 in threshold_factors1:
                for threshold_factor2 in threshold_factors2:
                    # print(f'{train_cell_size}, {guard_cell_size}, {threshold_factor1}, {threshold_factor2}') # for debug
                    coord_type = 'polar' if p2c else 'cart'
                    save_path = os.path.join(kradar_root, seq, 'radar', f'{coord_type}_obj_cacfar_{train_cell_size}_{guard_cell_size}_{int(threshold_factor1*100)}_{int(threshold_factor2*100)}')
                    os.makedirs(save_path, exist_ok=True)
                    save_path = os.path.join(save_path, f'{frame}.pcd')
                    xyz_coords = []
                    for obj in objs:
                        obj_pos = [obj['psr']['position']['x'], obj['psr']['position']['y'], obj['psr']['position']['z']]
                        obj_roi_half_length = (closest_odd_number(max(obj['psr']['scale'].values()) / grid_size) - 1) // 2
                        idx_x, idx_y, idx_z = np.argmin(np.abs(obj_pos[0] - arr_x_cb)), np.argmin(np.abs(obj_pos[1] - arr_y_cb)), np.argmin(np.abs(obj_pos[2] - arr_z_cb))
                        arr_x_cb_obj = arr_x_cb[idx_x-obj_roi_half_length:idx_x+obj_roi_half_length+1]
                        arr_y_cb_obj = arr_y_cb[idx_y-obj_roi_half_length:idx_y+obj_roi_half_length+1]
                        arr_z_cb_obj = arr_z_cb[idx_z-obj_roi_half_length:idx_z+obj_roi_half_length+1]
                        idx_x += pad_length
                        idx_y += pad_length
                        idx_z += pad_length
                        obj_roi_half_length += pad_length
                        rdr_tensor_obj = padded_rdr_tensor[idx_z-obj_roi_half_length:idx_z+obj_roi_half_length+1,\
                                    idx_y-obj_roi_half_length:idx_y+obj_roi_half_length+1, \
                                        idx_x-obj_roi_half_length:idx_x+obj_roi_half_length+1]
                        obj_xyz_coords = apply_cfar_torch(rdr_tensor_obj, kernel_3D, \
                                                    threshold_factor1, threshold_factor2, \
                                                        arr_x_cb_obj, arr_y_cb_obj, arr_z_cb_obj, pad_length, p2c, add_padding=False)
                        xyz_coords.append(obj_xyz_coords)
                    if len(xyz_coords) == 0:
                        continue
                    xyz_coords = np.concatenate(xyz_coords, axis=0)
                    point_cloud = o3d.geometry.PointCloud()
                    point_cloud.points = o3d.utility.Vector3dVector(xyz_coords)
                    o3d.io.write_point_cloud(save_path, point_cloud)
                    return True

if __name__ == '__main__':
    rdr_lidar_frame_difference = get_rdr_lidar_frame_difference(kradar_root, os.listdir('/mnt/ssd1/kradar_dataset/radar_tensor_zyx'))
    
    # --- debug --- 
    # rdr_path = '/mnt/ssd1/kradar_dataset/radar_tensor/11/radar_DEAR_D_downsampled_2/tesseract_00034.npy'
    # pick_rate(rdr_path, kradar_root, 33, True, arr_range, arr_azimuth, arr_elevation)
    # rdr_path = '/mnt/ssd1/kradar_dataset/radar_tensor_zyx/11/cube_00034.npy'
    # pick_rate(rdr_path, kradar_root, 33, False, arr_x_cb, arr_y_cb, arr_z_cb, pick_range=[0.001, 0.01], step=0.001)
    # cfar(rdr_path, kradar_root, 33, True, arr_range, arr_azimuth, arr_elevation)
    # --- debug --- 


    # CFAR
    # for seq in sorted(os.listdir('/mnt/ssd1/kradar_dataset/radar_tensor_zyx')):
    #     seq_path = os.path.join('/mnt/ssd1/kradar_dataset/radar_tensor_zyx', seq)
    #     print(f'Now processing {seq_path}')
    #     for rdr_path in tqdm(sorted(os.listdir(seq_path))):
    #         rdr_path = os.path.join(seq_path, rdr_path)
    #         cfar(rdr_path, kradar_root, rdr_lidar_frame_difference[seq], False, arr_x_cb, arr_y_cb, arr_z_cb, \
    #             train_cell_sizes=[30], guard_cell_sizes=[5], \
    #                 threshold_factors1=[1.5], threshold_factors2=[2.])

    # Pick top polar
    # for seq in sorted(os.listdir('/mnt/ssd1/kradar_dataset/radar_tensor')):
    #     seq_path = os.path.join('/mnt/ssd1/kradar_dataset/radar_tensor', seq, 'radar_DEAR_D_downsampled_2')
    #     print(f'Now processing {seq_path}')
    #     for rdr_path in tqdm(sorted(os.listdir(seq_path))):
    #         rdr_path = os.path.join(seq_path, rdr_path)
    #         pick_rate(rdr_path, kradar_root, rdr_lidar_frame_difference[seq], True, arr_range, arr_azimuth, arr_elevation, [0.05, 0.06])

    # Object CFAR
    for seq in sorted(os.listdir('/mnt/ssd1/kradar_dataset/radar_tensor_zyx')):
        seq_path = os.path.join('/mnt/ssd1/kradar_dataset/radar_tensor_zyx', seq)
        print(f'Now processing {seq_path}')
        for rdr_path in tqdm(sorted(os.listdir(seq_path))):
            rdr_path = os.path.join(seq_path, rdr_path)
            success = object_cfar(rdr_path, kradar_root, rdr_lidar_frame_difference[seq], False, arr_x_cb, arr_y_cb, arr_z_cb, \
                train_cell_sizes=[20], guard_cell_sizes=[5], \
                    threshold_factors1=[1.5], threshold_factors2=[1.5])
            if success:
                break