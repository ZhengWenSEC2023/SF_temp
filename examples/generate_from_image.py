"""
9. 19. 2020
by Zheng Wen

This file is used to generate 3D models from input 4-channel images

Run from anaconda console
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

BATCH_SIZE = 100
IMAGE_SIZE = 64
CLASS_IDS_ALL = (
        '02691156,02828884,02933112,02958343,03001627,03211117,03636649,' +
        '03691459,04090263,04256520,04379243,04401088,04530566')

PRINT_FREQ = 100
SAVE_FREQ = 100

MODEL_DIRECTORY = '/mnt/zhengwen/model_synthesis/SoftRas/data/results/models/checkpoint_0210000.pth.tar'
DATASET_DIRECTORY = '/mnt/zhengwen/model_synthesis/SoftRas/data/datasets'

SIGMA_VAL = 0.01
IMAGE_PATH = ''

# arguments

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

# setup model & optimizer
model = models.Model('/mnt/zhengwen/model_synthesis/SoftRas/data/obj/sphere/sphere_642.obj', args=args)
model = model.cuda()

state_dicts = torch.load(args.model_directory)
model.load_state_dict(state_dicts['model'], strict=False)
model.eval()

directory_output = '/mnt/zhengwen/model_synthesis/photo_from_life/123'
os.makedirs(directory_output, exist_ok=True)
directory_mesh = os.path.join(directory_output, args.experiment_id)
os.makedirs(directory_mesh, exist_ok=True)

IMG_PATH = '/mnt/zhengwen/model_synthesis/photo_from_life/texture'


end = time.time()

batch_time = AverageMeter()
data_time = AverageMeter()
losses = AverageMeter()
losses1 = AverageMeter()

iou_all = []

images = []

img_list = sorted(os.listdir(IMG_PATH))

for img_name in img_list:
    img = PIL.Image.open(os.path.join(IMG_PATH, img_name))
    img = np.asanyarray(img)
    images.append(img)

images = np.array(images)
images = images.transpose((0, 3, 1, 2))
images = np.ascontiguousarray(images)
images = torch.from_numpy(images.astype('float32') / 255.)
images = torch.autograd.Variable(images).cuda()

vertices, faces = model.reconstruct(images)
for k in range(len(img_list)):
    print(k)
    mesh_path = os.path.join(directory_output, img_list[k][:-4] + ".obj")
    srf.save_obj(mesh_path, vertices[k], faces[k])
