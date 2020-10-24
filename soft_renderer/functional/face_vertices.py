import torch


def face_vertices(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3] stores the actual location of each vertices
    :param faces: [batch size, number of faces, 3]       stores the index of 3 vertices for each faces
    :return: [batch size, number of faces, 3, 3]         return the index of 3 vertices and its 3D location [BS, # faces, index of vertices, location of vertices]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)


    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None] # change the index of faces starting from each .obj separately to total index
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]  # return each location
