"""
Generate datasets for the other several categories.
"""

import torch
import torch.nn.parallel
import examples.recon.datasets as datasets
from examples.recon.utils import AverageMeter, img_cvt
import soft_renderer as sr
import soft_renderer.functional as srf
import examples.recon.models as models
import time
import os
import imageio
import numpy as np
import PIL
import shutil

BATCH_SIZE = 100
IMAGE_SIZE = 64
PRINT_FREQ = 100
SAVE_FREQ = 100
MODEL_DIRECTORY = '/mnt/zhengwen/model_synthesis/SoftRas/data/results/models/checkpoint_0210000.pth.tar'
DATASET_DIRECTORY = '/mnt/zhengwen/model_synthesis/SoftRas/data/datasets'
SIGMA_VAL = 0.01
IMAGE_PATH = ''

SEQUENCE_RENDERING_STORE = (
    'rendered_0.png', 'rendered_23.png', 'rendered_22.png', 'rendered_21.png', 'rendered_20.png', 'rendered_19.png',
    'rendered_18.png', 'rendered_17.png', 'rendered_16.png', 'rendered_15.png', 'rendered_14.png', 'rendered_13.png',
    'rendered_12.png', 'rendered_11.png', 'rendered_10.png', 'rendered_9.png', 'rendered_8.png', 'rendered_7.png',
    'rendered_6.png', 'rendered_5.png', 'rendered_4.png', 'rendered_3.png', 'rendered_2.png', 'rendered_1.png'
)

CLASS_IDS_ALL = (
    '02691156', '02828884', '02933112', '02958343', '03001627', '03211117', '03636649',
    '03691459', '04090263', '04256520', '04379243', '04401088', '04530566')

BROKEN_ONJ = (
    '/mnt/zhengwen/model_synthesis/shapeNetCore/ShapeNetCore.v1/03624134/000000__broken__67ada28ebc79cc75a056f196c127ed77'
    '/mnt/zhengwen/model_synthesis/shapeNetCore/ShapeNetCore.v1/04074963/000000__broken__b65b590a565fa2547e1c85c5c15da7fb'
)


class Args:
    experiment_id = 'Sept_18_2020'
    model_directory = MODEL_DIRECTORY
    dataset_directory = DATASET_DIRECTORY
    class_ids = CLASS_IDS_ALL
    image_size = IMAGE_SIZE
    batch_size = BATCH_SIZE
    image_path = IMAGE_PATH
    sigma_val = SIGMA_VAL
    print_freq = PRINT_FREQ
    save_freq = SAVE_FREQ

args = Args()

model = models.Model('/mnt/zhengwen/model_synthesis/SoftRas/data/obj/sphere/sphere_642.obj', args=args)
model = model.cuda()

state_dicts = torch.load(args.model_directory)
model.load_state_dict(state_dicts['model'], strict=False)
model.eval()


category_to_num = {}
category_to_IoU = {}
category_to_argmax = {}
category_to_argmin = {}
category_to_max = {}
category_to_min = {}
category_to_mean = {}

RENDER_IMAGE_NAME = 'rendered'
camera_distance = 2.732
elevation = 30
azimuth = 0

root_dir = r'/mnt/zhengwen/model_synthesis/shapeNetCore/ShapeNetCore.v1'
save_dir = r'/mnt/zhengwen/model_synthesis/generated_model_9_19_2020'


sub_root_count = 0

for sub_root_dir in sorted(os.listdir(root_dir)):
    if sub_root_dir == '04090263':
        # continue

    # if os.path.isdir(os.path.join(root_dir, sub_root_dir)) and (sub_root_dir not in CLASS_IDS_ALL):

        npz_models = []
        npz_images = []

        category_to_num[sub_root_dir] = 0
        category_to_IoU[sub_root_dir] = []
        obj_count = 0
        for obj_dir in sorted(os.listdir(os.path.join(root_dir, sub_root_dir))):

            if os.path.join(os.path.join(os.path.join(root_dir, sub_root_dir)), obj_dir) in BROKEN_ONJ:
                continue

            print(sub_root_count, obj_count)

            # rendered_images_i = []
            #
            # for file_name in SEQUENCE_RENDERING_STORE:
            #     image = PIL.Image.open(os.path.join(os.path.join(os.path.join(os.path.join(root_dir, sub_root_dir)), obj_dir), file_name))
            #     image = image.resize((64, 64))
            #     image = np.asanyarray(image)
            #     # print(os.path.join(os.path.join(os.path.join(os.path.join(root_dir, sub_root_dir)), obj_dir), file_name))
            #     rendered_images_i.append(image)
            # rendered_images_i = np.array(rendered_images_i)
            # rendered_images_i = rendered_images_i.transpose((0, 3, 1, 2))
            #
            # npz_images.append(rendered_images_i)

            obj_file_i = os.path.join(os.path.join(os.path.join(os.path.join(root_dir, sub_root_dir)), obj_dir),
                                      'model.obj')
            mesh = sr.Mesh.from_obj(obj_file_i)

            vertices_real, faces_real = mesh.vertices, mesh.faces

            faces_real_ = srf.face_vertices(vertices_real, faces_real).data
            faces_real_norm = faces_real_ * 1. * (32. - 1) / 32. + 0.5
            voxels_real = srf.voxelization(faces_real_norm, 32, False).cpu().numpy()

            npz_models.append(np.squeeze(voxels_real).transpose((1, 2, 0)))

            obj_count += 1

            if obj_count == 10:
                break

        npz_models = np.array(npz_models)
        npz_images = np.array(npz_images)

        sub_root_count += 1
        idx = np.random.permutation(len(npz_models))
        train_idx = idx[: int(len(npz_models) * 0.7)]
        val_idx = idx[int(len(npz_models) * 0.7) + 1: int(len(npz_models) * 0.8)]
        test_idx = idx[int(len(npz_models) * 0.8) + 1:]

        # np.savez(os.path.join('/mnt/zhengwen/model_synthesis/SoftRas/data/whole_dataset', sub_root_dir + "_train_images"), npz_images[train_idx])
        # np.savez(os.path.join('/mnt/zhengwen/model_synthesis/SoftRas/data/whole_dataset', sub_root_dir + "_val_images"), npz_images[val_idx])
        # np.savez(os.path.join('/mnt/zhengwen/model_synthesis/SoftRas/data/whole_dataset', sub_root_dir + "_test_images"), npz_images[test_idx])
        # np.savez(os.path.join('/mnt/zhengwen/model_synthesis/SoftRas/data/whole_dataset', sub_root_dir + "_train_voxels"), npz_models[train_idx])
        # np.savez(os.path.join('/mnt/zhengwen/model_synthesis/SoftRas/data/whole_dataset', sub_root_dir + "_val_voxels"), npz_models[val_idx])
        # np.savez(os.path.join('/mnt/zhengwen/model_synthesis/SoftRas/data/whole_dataset', sub_root_dir + "_test_voxels"), npz_models[test_idx])
        # np.savez(os.path.join('/mnt/zhengwen/model_synthesis/SoftRas', sub_root_dir + "_train_images"), npz_images[train_idx])
        # np.savez(os.path.join('/mnt/zhengwen/model_synthesis/SoftRas', sub_root_dir + "_val_images"), npz_images[val_idx])
        # np.savez(os.path.join('/mnt/zhengwen/model_synthesis/SoftRas', sub_root_dir + "_test_images"), npz_images[test_idx])
        np.savez(os.path.join('/mnt/zhengwen/model_synthesis/SoftRas', sub_root_dir + "_train_voxels"), npz_models[train_idx])
        np.savez(os.path.join('/mnt/zhengwen/model_synthesis/SoftRas', sub_root_dir + "_val_voxels"), npz_models[val_idx])
        np.savez(os.path.join('/mnt/zhengwen/model_synthesis/SoftRas', sub_root_dir + "_test_voxels"), npz_models[test_idx])

        break

