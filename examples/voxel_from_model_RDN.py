"""
Render the models from 24 elevation angles, as in thesis NMR
Save as an image.

9. 17. 2020
created by Zheng Wen

9. 19. 2020
ALL RENDER ARE FINISHED WITHOUT TEXTURE

Run from anaconda console

NOTE:
    RENDER FROM ORIGINAL SHOULD BE RANGE(360, 0, -15)
    HERE RANGE(0, 360, 15)

    SOLUTION: RENAME FILES OR GENERATE DATASETS IN FOLLOWING SEQUENCES:
        0, 23, 22, ..., 1

"""


import matplotlib.pyplot as plt
import os
import tqdm
import numpy as np
import imageio

import soft_renderer as sr

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

CLASS_IDS_ALL = (
    # "02958343",  # car
    "02691156",  # airplane
    # "03001627",  # chair
)

RENDERED_CLASS = ()

BROKEN_ONJ = (
    '/mnt/zhengwen/model_synthesis/shapeNetCore/ShapeNetCore.v1/03624134/000000__broken__67ada28ebc79cc75a056f196c127ed77'
    '/mnt/zhengwen/model_synthesis/shapeNetCore/ShapeNetCore.v1/04074963/000000__broken__b65b590a565fa2547e1c85c5c15da7fb'
)

category_to_num = {}

RENDER_IMAGE_NAME_RGB = 'RGB'
RENDER_IMAGE_NAME_D = 'depth'
RENDER_IMAGE_NAME_NORMAL = 'normal'
camera_distance = 3.5
elevation = 30
azimuth = 0

root_dir = r'/mnt/zhengwen/model_synthesis/shapeNetCore/ShapeNet_OCT_29/ShapeNetCore.v1'
sub_root_count = 0
for sub_root_dir in sorted(os.listdir(root_dir)):

    if os.path.isdir(os.path.join(root_dir, sub_root_dir)) and (sub_root_dir in CLASS_IDS_ALL) and (sub_root_dir not in RENDERED_CLASS):
        category_to_num[sub_root_dir] = 0
        obj_count = 0

        voxs = []

        for obj_dir in sorted(os.listdir(os.path.join(root_dir, sub_root_dir))):

            print(sub_root_count, obj_count)

            obj_file_i = os.path.join(os.path.join(os.path.join(os.path.join(root_dir, sub_root_dir)), obj_dir),
                                      'model.obj')
            if os.path.join(os.path.join(os.path.join(root_dir, sub_root_dir)), obj_dir) in BROKEN_ONJ:
                continue

            mesh = sr.Mesh.from_obj(obj_file_i)
            vox = mesh.voxelize(64)

            obj_count += 1

            voxs.append(np.array(vox.detach().cpu()))

        sub_root_count += 1
        voxs = np.concatenate(voxs, axis=0)
        np.save(sub_root_dir + "_voxel.npz", voxs)
