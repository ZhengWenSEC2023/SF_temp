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

RENDER_IMAGE_NAME_RGB = 'RGB'
RENDER_IMAGE_NAME_D = 'depth'
RENDER_IMAGE_NAME_NORMAL = 'normal'
camera_distance = 10
elevation = 30
azimuth = 0

obj_file_i = os.path.join("/mnt/zhengwen/model_synthesis/SF_temp/data/obj/sphere/sphere_642.obj")
img_file_rgb = "/mnt/zhengwen/model_synthesis/SF_temp/data/obj/sphere/sphere_642.obj" + RENDER_IMAGE_NAME_RGB
img_file_depth = "/mnt/zhengwen/model_synthesis/SF_temp/data/obj/sphere/sphere_642.obj" + RENDER_IMAGE_NAME_D
img_file_normal = "/mnt/zhengwen/model_synthesis/SF_temp/data/obj/sphere/sphere_642.obj" + RENDER_IMAGE_NAME_NORMAL
mesh = sr.Mesh.from_obj(obj_file_i)
renderer = sr.SoftRenderer(camera_mode='look_at')

for azimuth in range(0, 360, 15):
    count = azimuth // 15
    # rest mesh to initial state
    mesh.reset_()
    renderer.transform.set_eyes_from_angles(camera_distance, elevation, azimuth)
    images = renderer.render_mesh(mesh)

    image_rgb = images[0].detach().cpu().numpy()[0]
    imageio.imwrite(img_file_rgb + '_' + str(count) + '.png', (255 * image_rgb[:, 128 - 32: 128 + 32, 128 - 32: 128 + 32]).transpose((1, 2, 0)).astype(np.uint8))

    image_d = images[1].detach().cpu().numpy()[0]
    image_d[image_d != 0] = 2 * 1 / image_d[image_d != 0]

    imageio.imwrite(img_file_depth + '_' + str(count) + '.png', (255 * image_d[:, 128 - 32: 128 + 32, 128 - 32: 128 + 32]).transpose((1, 2, 0)).astype(np.uint8))

    image_normal = images[2].detach().cpu().numpy()[0]
    imageio.imwrite(img_file_normal + '_' + str(count) + '.png', (255 * image_normal[:, 128 - 32: 128 + 32, 128 - 32: 128 + 32]).transpose((1, 2, 0)).astype(np.uint8))


