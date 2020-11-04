import numpy as np
import os

depth = np.load(os.path.join("/mnt/zhengwen/model_synthesis/SF_temp", "02691156_depth.npz.npy"))
normal = np.load(os.path.join("/mnt/zhengwen/model_synthesis/SF_temp", "02691156_norma.npz.npy"))
rgb = np.load(os.path.join("/mnt/zhengwen/model_synthesis/SF_temp", "02691156_rgb.npz.npy"))
voxel = np.load(os.path.join("/mnt/zhengwen/model_synthesis/SF_temp", "02691156_voxel.npz.npy"))

num_train = 7 * depth.shape[0] // 10
num_val = 8 * depth.shape[0] // 10
num_test = depth.shape[0]

depth_train = depth[0: num_train]
depth_val = depth[num_train: num_val]
depth_test = depth[num_val: num_test]

normal_train = normal[0: num_train]
normal_val = normal[num_train: num_val]
normal_test = normal[num_val: num_test]

rgba_train = rgb[0: num_train]
rgba_val = rgb[num_train: num_val]
rgba_test = rgb[num_val: num_test]

voxel_train = voxel[0: num_train]
voxel_val = voxel[num_train: num_val]
voxel_test = voxel[num_val: num_test]

np.savez(os.path.join("/mnt/zhengwen/model_synthesis/SF_temp", "02691156_depth_train.npz"), depth_train)
np.savez(os.path.join("/mnt/zhengwen/model_synthesis/SF_temp", "02691156_depth_val.npz"), depth_val)
np.savez(os.path.join("/mnt/zhengwen/model_synthesis/SF_temp", "02691156_depth_test.npz"), depth_test)

np.savez(os.path.join("/mnt/zhengwen/model_synthesis/SF_temp", "02691156_norma_train.npz"), normal_train)
np.savez(os.path.join("/mnt/zhengwen/model_synthesis/SF_temp", "02691156_norma_val.npz"), normal_val)
np.savez(os.path.join("/mnt/zhengwen/model_synthesis/SF_temp", "02691156_norma_test.npz"), normal_test)

np.savez(os.path.join("/mnt/zhengwen/model_synthesis/SF_temp", "02691156_rgba_train.npz"), rgba_train)
np.savez(os.path.join("/mnt/zhengwen/model_synthesis/SF_temp", "02691156_rgba_val.npz"), rgba_val)
np.savez(os.path.join("/mnt/zhengwen/model_synthesis/SF_temp", "02691156_rgba_test.npz"), rgba_test)

np.savez(os.path.join("/mnt/zhengwen/model_synthesis/SF_temp", "02691156_voxel_train.npz"), voxel_train)
np.savez(os.path.join("/mnt/zhengwen/model_synthesis/SF_temp", "02691156_voxel_val.npz"), voxel_val)
np.savez(os.path.join("/mnt/zhengwen/model_synthesis/SF_temp", "02691156_voxel_test.npz"), voxel_test)

