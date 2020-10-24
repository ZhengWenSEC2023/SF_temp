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
    if os.path.isdir(os.path.join(root_dir, sub_root_dir)) and (sub_root_dir not in CLASS_IDS_ALL):
        category_to_num[sub_root_dir] = 0
        category_to_IoU[sub_root_dir] = []
        obj_count = 0
        for obj_dir in sorted(os.listdir(os.path.join(root_dir, sub_root_dir))):

            print(sub_root_count, obj_count)

            obj_file_i = os.path.join(os.path.join(os.path.join(os.path.join(root_dir, sub_root_dir)), obj_dir),
                                      'model.obj')
            if os.path.join(os.path.join(os.path.join(root_dir, sub_root_dir)), obj_dir) in BROKEN_ONJ:
                continue

            rendered_images_i = []
            for file_name in sorted(os.listdir(os.path.join(os.path.join(os.path.join(root_dir, sub_root_dir)), obj_dir))):
                if 'rendered' in file_name:
                    image = PIL.Image.open(os.path.join(os.path.join(os.path.join(os.path.join(root_dir, sub_root_dir)), obj_dir), file_name))
                    image = image.resize((64, 64))
                    image = np.asanyarray(image)
                    # print(os.path.join(os.path.join(os.path.join(os.path.join(root_dir, sub_root_dir)), obj_dir), file_name))
                    rendered_images_i.append(image)
            rendered_images_i = np.array(rendered_images_i)
            num_rendered_images = rendered_images_i.shape[0]
            rendered_images_i = rendered_images_i.transpose((0, 3, 1, 2))
            rendered_images_i = np.ascontiguousarray(rendered_images_i)
            rendered_images_i = torch.from_numpy(rendered_images_i.astype('float32') / 255.)
            rendered_images_i = torch.autograd.Variable(rendered_images_i).cuda()

            mesh = sr.Mesh.from_obj(obj_file_i)

            vertices_real, faces_real = mesh.vertices, mesh.faces

            vertices_real = [vertices_real for _ in range(num_rendered_images)]
            faces_real = [faces_real for _ in range(num_rendered_images)]

            vertices_real = torch.cat(vertices_real, dim=0)
            faces_real = torch.cat(faces_real, dim=0)

            vertices_generate, faces_generate = model.reconstruct(rendered_images_i)

            # srf.save_obj('/mnt/zhengwen/model_synthesis/shapeNetCore/test_generate.obj', vertices_generate[0], faces_generate[0])
            # srf.save_obj('/mnt/zhengwen/model_synthesis/shapeNetCore/test_real.obj', vertices_real[0], faces_real[0])

            faces_real_ = srf.face_vertices(vertices_real, faces_real).data
            faces_real_norm = faces_real_ * 1. * (32. - 1) / 32. + 0.5
            voxels_real = srf.voxelization(faces_real_norm, 32, False).cpu().numpy()

            faces_generate_ = srf.face_vertices(vertices_generate, faces_generate).data
            faces_generate_norm = faces_generate_ * 1. * (32. - 1) / 32. + 0.5
            voxels_generate = srf.voxelization(faces_generate_norm, 32, False).cpu().numpy()
            voxels_generate = voxels_generate.transpose(0, 2, 1, 3)[:, :, :, ::-1]

            iou = (voxels_real * voxels_generate).sum((1, 2, 3)) / (0 < (voxels_real + voxels_generate)).sum((1, 2, 3))
            category_to_IoU[sub_root_dir].append(np.mean(iou))
            obj_count += 1
    sub_root_count += 1

max_IoU = 0
min_IoU = 1
for each_key in category_to_IoU:
    max_IoU = max(max(category_to_IoU[each_key]), max_IoU)
    min_IoU = min(min(category_to_IoU[each_key]), min_IoU)

max_idx = 0
min_idx = 0
for each_key in category_to_IoU:
    for i in range(len(category_to_IoU[each_key])):
        if category_to_IoU[each_key][i] == max_IoU:
            max_idx = (each_key, sorted(os.listdir(os.path.join(root_dir, each_key)))[i], max_IoU)
        if category_to_IoU[each_key][i] == min_IoU:
            min_idx = (each_key, sorted(os.listdir(os.path.join(root_dir, each_key)))[i], min_IoU)


for each_key in category_to_IoU:
    category_to_argmax[each_key] = np.argmax(category_to_IoU[each_key])
    category_to_argmin[each_key] = np.argmin(category_to_IoU[each_key])
    category_to_max[each_key] = np.max(category_to_IoU[each_key])
    category_to_min[each_key] = np.min(category_to_IoU[each_key])
    category_to_mean[each_key] = np.mean(category_to_IoU[each_key])
    category_to_num[each_key] = len(category_to_IoU[each_key])

for each_key in category_to_IoU:
    save_gen_model_path = os.path.join(save_dir, each_key + '_generated_model.obj')
    save_ori_model_path = os.path.join(save_dir, each_key + '_original_model.obj')

    file_list = sorted(os.listdir(os.path.join(root_dir, each_key)))
    if '000000__broken__67ada28ebc79cc75a056f196c127ed77' in file_list:
        file_list.remove('000000__broken__67ada28ebc79cc75a056f196c127ed77')
    if '000000__broken__b65b590a565fa2547e1c85c5c15da7fb' in file_list:
        file_list.remove('000000__broken__b65b590a565fa2547e1c85c5c15da7fb')

    image_path = os.path.join(root_dir, each_key, file_list[category_to_argmin[each_key]], 'rendered_0.png')
    original_model_path = os.path.join(root_dir, each_key, file_list[category_to_argmin[each_key]], 'model.obj')
    image = PIL.Image.open(image_path)
    image = image.resize((64, 64))
    image = np.asanyarray(image)[None, :, :, :]
    image = image.transpose((0, 3, 1, 2))
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image.astype('float32') / 255.)
    image = torch.autograd.Variable(image).cuda()
    vertice_generate, face_generate = model.reconstruct(image)
    srf.save_obj(save_gen_model_path, vertice_generate[0], face_generate[0])
    shutil.copy(original_model_path, save_ori_model_path)
    print(each_key, category_to_num[each_key], category_to_max[each_key], category_to_min[each_key], category_to_mean[each_key])