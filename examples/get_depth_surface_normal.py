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


def toBaryCentric(U_inv, i, j):
    """
    :param:
        U_inv: inverse metrix of transformation
        i, j index of current pixel
    :return:
        return the baryCentric coordinate defined by fi
    """
    pi = np.array([i, j, 1])
    return np.matmul(U_inv, pi)


def calculateMeshInv(mesh_face_vertices):
    """
    calculate the inverse matrix for each mesh to convert the Cartesian coordinates to barycentric coordinates defined by
    the certain mesh.
    Just to save the time of computing

    :param:
        mesh_face_vertices: mesh list with [number of mesh, vertices index, (0: x, 1: y, 2: z)]
    :return:
        [number of vertices, U^(-1)]
    """
    mesh_inv = []
    for mesh in mesh_face_vertices:
        U = np.array([
            [mesh[0, 0], mesh[1, 0], mesh[2, 0]],
            [mesh[0, 1], mesh[1, 1], mesh[2, 1]],
            [1, 1, 1],
        ])
        mesh_inv.append(np.linalg.inv(U))
    return np.array(mesh_inv)


def out_box(i, j, mesh):
    """
    Judge whether current pixel lies inside the bounding box formed by the vertices of current mesh

    :param i: idx of pixel
    :param j: idx of pixel
    :param mesh: current mesh ( [ [x0, y0, z0], [x1, y1, z1], [x2, y2, z2] ] )
    :return:
        bool: in box: true, else false
    """
    return (i > max(max(mesh[0, 0], mesh[1, 0]), mesh[2, 0]) or
            i < min(min(mesh[0, 0], mesh[1, 0]), mesh[2, 0]) or
            j > max(max(mesh[0, 1], mesh[1, 1]), mesh[2, 1]) or
            j < min(min(mesh[0, 1], mesh[1, 1]), mesh[2, 1]))


def calculateMeshNormal(mesh_face_vertices):
    """
    calculate the normal vector for each mesh,
    x: [-1, 1] -> [0, 255]
    y: [-1, 1] -> [0, 255]
    z: [ 0, 1] -> [128, 255]  (z should be also [-1, 1] -> [0, 255], while only the outside would be shown, but also calculated in this way)
    :param:
        mesh_face_vertices: mesh list with [number of mesh, vertices index, (0: x, 1: y, 2: z)]
    :return:
        [number of vertices, 3] -> xyz
    """
    mesh_normal = []
    for mesh in mesh_face_vertices:
        v1x = mesh[1, 0] - mesh[0, 0]
        v1y = mesh[1, 1] - mesh[0, 1]
        v1z = mesh[1, 2] - mesh[0, 2]
        v2x = mesh[2, 0] - mesh[1, 0]
        v2y = mesh[2, 1] - mesh[1, 1]
        v2z = mesh[2, 2] - mesh[1, 2]
        
        normal = np.array([v1y * v2z - v1z * v2y, v1z * v2x - v1x * v2z, v1x * v2y - v1y * v2x])
        normal = normal / np.max((np.linalg.norm(normal), 1e-5))
        normal = (normal + 1) * 127.5
        mesh_normal.append(normal)
    return np.array(mesh_normal)


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

camera_distance = 2.732
elevation = 30
azimuth = 90
image_size = 256
# reference: https://github.com/mrdoob/three.js/blob/dae1116271de869672a44dd705b76c09fc135f48/src/cameras/PerspectiveCamera.js
# suppose size of film to be 256 / 4 (mm)
# field of view = 30 degree (BY DEFAULT)

# relationship between (focal length, film size, field of view)
# focal length = 0.5 * film_size / tan(0.5 * fov)
# film size is the edge length of the film, height or weight (suppose a square film), fov is the field of view,
# in terms of degree.

film_size = 256 / 4

root_dir = r'/mnt/zhengwen/model_synthesis/shapeNetCore/ShapeNetCore.v1'
sub_root_count = 0

mesh = sr.Mesh.from_obj(r'/mnt/zhengwen/model_synthesis/shapeNetCore/ShapeNetCore.v1/02691156/ffbc31352a34c3e1ffb94dfdd6ddfaf0/model.obj')
renderer = sr.SoftRenderer(camera_mode='look_at')
# rest mesh to initial state
mesh.reset_()
renderer.transform.set_eyes_from_angles(camera_distance, elevation, azimuth)
renderer.set_texture_mode(mesh.texture_type)
# renderer.render_mesh(mesh)


mesh = renderer.lighting(mesh)
mesh = renderer.transform(mesh)
mesh_face_vertices = mesh.face_vertices.cpu().numpy()[0]

z_near = 1
z_far = 100

a = -(z_far + z_near) / (z_far - z_near)
b = (-2 * z_far * z_near / (z_far - z_near))

# finish clip transform
mesh_face_vertices[:, :, 2] = -a + b / mesh_face_vertices[:, :, 2]

mesh_face_vertices_before_viewport = mesh_face_vertices.copy()

# viewport transform
v_r, v_l = 256, 0
v_t, v_b = 256, 0

mesh_face_vertices[:, :, 0] = ((v_r - v_l) / 2) * mesh_face_vertices[:, :, 0] + ((v_r + v_l) / 2)
mesh_face_vertices[:, :, 1] = ((v_t - v_b) / 2) * mesh_face_vertices[:, :, 1] + ((v_t - v_b) / 2)
mesh_face_vertices[:, :, 2] = 1/2 * mesh_face_vertices[:, :, 2] + 1/2

INF_DEPTH = 2
depth = INF_DEPTH * np.ones((256, 256))
normal_vector = np.zeros((256, 256, 3))
# print(np.max(mesh_face_vertices[:, :, 0]), np.min(mesh_face_vertices[:, :, 0]))
# print(np.max(mesh_face_vertices[:, :, 1]), np.min(mesh_face_vertices[:, :, 1]))
# print(np.max(mesh_face_vertices[:, :, 2]), np.min(mesh_face_vertices[:, :, 2]))
# print()
# print(toBaryCentric(mesh_face_vertices[0], 138.68094, 129.82126))

inv_matrix = calculateMeshInv(mesh_face_vertices)
normal_vector_set = calculateMeshNormal(mesh_face_vertices_before_viewport)
EPL = 0.001
for i in range(128 - 32, 128 + 32):
    for j in range(128 - 32, 128 + 32):
        print(i, j)
        shallowest_mesh = 0
        flag_normal = False
        for k in range(mesh_face_vertices.shape[0]):
            if out_box(i, j, mesh_face_vertices[k]):
                continue
            pi = toBaryCentric(inv_matrix[k], i, j)
            if np.min(pi) - EPL < 0:
                continue   # out of mesh

            cur_depth = np.sum(abs(pi) * mesh_face_vertices[k, :, 2])
            if depth[i, j] > cur_depth:
                depth[i, j] = cur_depth
                shallowest_mesh = k
                flag_normal = True

        if flag_normal:
            normal_vector[i, j, 0] = normal_vector_set[shallowest_mesh, 0]
            normal_vector[i, j, 1] = normal_vector_set[shallowest_mesh, 1]
            normal_vector[i, j, 2] = normal_vector_set[shallowest_mesh, 2]

np.save("depth.npy", depth)
np.save("normal_vector.npy", normal_vector)
