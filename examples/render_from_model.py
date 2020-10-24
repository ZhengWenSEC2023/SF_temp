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

CLASS_IDS_ALL = (
    '02691156', '02828884', '02933112', '02958343', '03001627', '03211117', '03636649',
    '03691459', '04090263', '04256520', '04379243', '04401088', '04530566')

RENDERED_CLASS = ('02691156', '02747177', '02773838', '02801938', '02808440', '02818832', '02828884', '02834778',
                  '02843684', '02858304', '02871439', '02876657', '02880940', '02924116', '02933112', '02942699',
                  '02946921', '02954340', '02958343', '02992529', '03001627', '03046257', '03085013', '03207941',
                  '03211117', '03261776', '03325088', '03337140', '03467517', '03513137', '03593526', '03624134',
                  '03636649', '03642806', '03691459', '03710193', '03759954', '03761084', '03790512', '03797390',
                  '03928116', '03938244', '03948459', '03991062', '04004475',
                  )

BROKEN_ONJ = (
    '/mnt/zhengwen/model_synthesis/shapeNetCore/ShapeNetCore.v1/03624134/000000__broken__67ada28ebc79cc75a056f196c127ed77'
    '/mnt/zhengwen/model_synthesis/shapeNetCore/ShapeNetCore.v1/04074963/000000__broken__b65b590a565fa2547e1c85c5c15da7fb'
)

category_to_num = {}

RENDER_IMAGE_NAME = 'rendered'
camera_distance = 2.732
elevation = 30
azimuth = 0

root_dir = r'/mnt/zhengwen/model_synthesis/shapeNetCore/ShapeNetCore.v1'
sub_root_count = 0
for sub_root_dir in sorted(os.listdir(root_dir)):
    if os.path.isdir(os.path.join(root_dir, sub_root_dir)) and (sub_root_dir not in CLASS_IDS_ALL) and (sub_root_dir not in RENDERED_CLASS):
        category_to_num[sub_root_dir] = 0
        obj_count = 0
        for obj_dir in sorted(os.listdir(os.path.join(root_dir, sub_root_dir))):

            print(sub_root_count, obj_count)

            obj_file_i = os.path.join(os.path.join(os.path.join(os.path.join(root_dir, sub_root_dir)), obj_dir),
                                      'model.obj')
            if os.path.join(os.path.join(os.path.join(root_dir, sub_root_dir)), obj_dir) in BROKEN_ONJ:
                continue
            img_file_o = os.path.join(os.path.join(os.path.join(os.path.join(root_dir, sub_root_dir)), obj_dir),
                                      RENDER_IMAGE_NAME)
            mesh = sr.Mesh.from_obj(obj_file_i)
            renderer = sr.SoftRenderer(camera_mode='look_at')
            for azimuth in range(0, 360, 15):
                count = azimuth // 15
                # rest mesh to initial state
                mesh.reset_()
                renderer.transform.set_eyes_from_angles(camera_distance, elevation, azimuth)
                images = renderer.render_mesh(mesh)
                image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
                imageio.imwrite(img_file_o + '_' + str(count) + '.png', (255 * image).astype(np.uint8))
            obj_count += 1
    sub_root_count += 1
