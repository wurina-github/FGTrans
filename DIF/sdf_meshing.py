'''Create mesh from SDF
'''

import logging
import numpy as np
import plyfile
import skimage.measure
import time
import torch
from collections import OrderedDict

'''Adapted from the DeepSDF repository https://github.com/facebookresearch/DeepSDF
'''

def create_mesh(model, filename,category, subject_idx=0, embedding=None, N=128, max_batch=64 ** 3, offset=None, scale=None,level=0.0, get_color=True,
                ):
    start = time.time()
    ply_filename = filename

    model.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    samples_34 = samples.clone()
    num_samples = N ** 3

    samples.requires_grad = False

    head = 0
    if embedding is None:
        subject_idx = torch.Tensor([subject_idx]).squeeze().long().cuda()[None,...]
        embedding = model.get_latent_code(subject_idx)

    if get_color is True:
        if category == 'bottle':
            subject_34 = torch.Tensor([10]).squeeze().long().cuda()[None, ...]
            embedding_34 = model.get_latent_code(subject_34)
        elif category == 'knife':
            subject_34 = torch.Tensor([20]).squeeze().long().cuda()[None, ...]
            embedding_34 = model.get_latent_code(subject_34)
    while head < num_samples:
        print(head)
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()[None,...]
        samples[head: min(head + max_batch, num_samples), 3] = (
            model.inference(sample_subset, embedding)
                .squeeze()  # .squeeze(1)
                .detach()
                .cpu()
        )
        if get_color is True:
            sample_subset_34 = samples_34[head : min(head + max_batch, num_samples), 0:3].cuda()[None,...]
            samples_34[head : min(head + max_batch, num_samples), 3] = (
                model.inference(sample_subset_34,embedding_34)
                .squeeze()#.squeeze(1)
                .detach()
                .cpu()
            )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)
    if get_color is True:
        sdf_values_34 = samples_34[:, 3]
        sdf_values_34 = sdf_values_34.reshape(N, N, N)

    end = time.time()
    print("sampling takes: %f" % (end - start))

    if not get_color:
        convert_sdf_samples_to_ply(
            sdf_values.data.cpu(),
            voxel_origin,
            voxel_size,
            ply_filename + ".ply",
            category,
            offset,
            scale,
            level
        )
    else:
        convert_sdf_samples_with_color_to_ply(
            model,
            embedding,
            embedding_34,
            sdf_values.data.cpu(),
            sdf_values_34.data.cpu(),
            voxel_origin,
            voxel_size,
            ply_filename + ".ply",
            category,
            offset,
            scale,
            level

        )


def get_mesh_color(mesh_points,mesh_points_34,embedding,embedding_34,model,category):

    model.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    # mesh_points.requires_grad = False
    # mesh_colors = np.zeros_like(mesh_points)
    mesh_coords = torch.zeros([mesh_points.shape[0], mesh_points.shape[1]], dtype=torch.float)
    mesh_coords_34 = torch.zeros([mesh_points_34.shape[0], mesh_points_34.shape[1]], dtype=torch.float)
    num_samples = mesh_points.shape[0]
    num_samples_34 = mesh_points_34.shape[0]

    print('num_samples',num_samples)
    max_batch=64 ** 3

    head = 0
    while head < num_samples_34:
        print(head)
        sample_subset = \
        torch.from_numpy(mesh_points_34[head: min(head + max_batch, num_samples_34), 0:3]).float().cuda()[None, ...]

        mesh_coords_34[head: min(head + max_batch, num_samples_34), 0:3] = (
            model.get_template_coords(sample_subset, embedding_34)
                .squeeze()  # .squeeze(1)
                .detach()
                .cpu()
        )

        head += max_batch

    head = 0
    while head < num_samples:
        print(head)
        sample_subset = torch.from_numpy(mesh_points[head : min(head + max_batch, num_samples), 0:3]).float().cuda()[None,...]

        mesh_coords[head : min(head + max_batch, num_samples), 0:3] = (
            model.get_template_coords(sample_subset,embedding)
            .squeeze()#.squeeze(1)
            .detach()
            .cpu()
        )

        head += max_batch

    print("begin cal dis:{}x{}".format(mesh_coords.shape, mesh_coords_34.shape))
    dis = (((mesh_coords.unsqueeze(1).expand(-1, mesh_coords_34.shape[0], -1) - mesh_coords_34.unsqueeze(0).expand(mesh_coords.shape[0], -1, -1)) ** 2).sum(-1).sqrt())
    print("end cal dis:{}x{}".format(mesh_coords.shape, mesh_coords_34.shape))
    idx = dis.argmin(-1)
    print(mesh_coords.shape)
    print(mesh_coords_34.shape)
    print(idx.shape)
    if category == 'knife':
        color_dict = np.load('./util/knife0020_color_seg.npy') #$$$ npy file + subject_idx
    elif category == 'bottle':
        color_dict = np.load('./util/bottle0010_color.npy') #$$$ npy file + subject_idx


    idx[idx > (color_dict.shape[0]-1)] = color_dict.shape[0]-1
    mesh_colors = color_dict[idx]

    # mesh_colors = np.clip(mesh_colors/2+0.5,0,1) # normalize color to 0-1
    mesh_colors = np.clip(mesh_colors/255,0,1) # normalize color to 0-1

    return mesh_colors


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    category,
    offset=None,
    scale=None,
    level=0.0,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
            numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3
        )
    except:
        pass

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )


def convert_sdf_samples_with_color_to_ply(
    model,
    embedding,
    embedding_34,
    pytorch_3d_sdf_tensor,
    pytorch_3d_sdf_tensor_34,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    cate,
    offset=None,
    scale=None,
    level=0.0,

):
    """
    Convert sdf samples to .ply with color-coded template coordinates

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    numpy_3d_sdf_tensor_34 = pytorch_3d_sdf_tensor_34.numpy()

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    verts_34, faces_34, normals_34, values_34 = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
            numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3
        )
        verts_34, faces_34, normals_34, values_34 = skimage.measure.marching_cubes_lewiner(
            numpy_3d_sdf_tensor_34, level=level, spacing=[voxel_size] * 3
        )
    except:
        pass

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]


    mesh_points_34 = np.zeros_like(verts_34)
    mesh_points_34[:, 0] = voxel_grid_origin[0] + verts_34[:, 0]
    mesh_points_34[:, 1] = voxel_grid_origin[1] + verts_34[:, 1]
    mesh_points_34[:, 2] = voxel_grid_origin[2] + verts_34[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset


    mesh_colors = get_mesh_color(mesh_points,mesh_points_34,embedding,embedding_34,model,cate)
    mesh_colors = np.clip(mesh_colors*255,0,255).astype(np.uint8)

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    colors_tuple = np.zeros((num_verts,), dtype=[("red", "u1"), ("green", "u1"), ("blue", "u1")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    for i in range(0, num_verts):
        colors_tuple[i] = tuple(mesh_colors[i, :])

    verts_all = np.empty(num_verts,verts_tuple.dtype.descr + colors_tuple.dtype.descr)

    for prop in verts_tuple.dtype.names:
        verts_all[prop] = verts_tuple[prop]

    for prop in colors_tuple.dtype.names:
        verts_all[prop] = colors_tuple[prop]


    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_all, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces],text=True)
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )