import functools
import trimesh
import numpy as np
import open3d as o3d
import sys
import os
import torch
import argparse
sys.path.append('./')

def read_off(object_dir, object_name):
    file = open(object_dir + '/' + object_name, 'r')
    if 'OFF' != file.readline().strip():
        raise ('Not a valid OFF header')
    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    verts = np.asarray(verts)
    try:
        faces = [[int(s) for s in file.readline().strip().split(' ')] for i_face in range(n_faces)]
        faces = np.asarray(faces)[:, 1:]
        return verts, faces
    except:
        return verts,None

def read_ply(object_dir, object_name):
    file = open(object_dir + '/' + object_name, 'r')
    if 'ply' != file.readline().strip():
        raise ('Not a valid OFF header')
    file.readline()
    _, _, n_verts = tuple([s for s in file.readline().strip().split(' ')])
    file.readline()
    file.readline()
    file.readline()
    _, _, n_faces = tuple([s for s in file.readline().strip().split(' ')])
    if n_faces == 'red':
        file.readline()
        file.readline()
        _, _, n_faces = tuple([s for s in file.readline().strip().split(' ')])
    n_verts, n_faces = int(n_verts), int(n_faces)
    file.readline()
    file.readline()

    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    verts = np.asarray(verts)[:, :3]
    faces = [[int(s) for s in file.readline().strip().split(' ')] for i_face in range(n_faces)]
    faces = np.asarray(faces)[:, 1:]
    return verts, faces


def show_point(points,pointsorg = None):
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5,origin=[0,0,0])
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.paint_uniform_color([1,0.706,0])  # 黄色
    if pointsorg is not None:
        axis_org = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5,origin=[0,0,0])
        point_cloud_org = o3d.geometry.PointCloud()
        point_cloud_org.points = o3d.utility.Vector3dVector(pointsorg)
        point_cloud_org.paint_uniform_color([1, 0, 0])  # 红色
        o3d.visualization.draw_geometries([axis, point_cloud,axis_org,point_cloud_org])
    else:
        o3d.visualization.draw_geometries([axis, point_cloud])


def scale_to_unit_sphere_dif(mesh,cate):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()
    elif cate == 'knife':
        vertices = mesh.vertices - (mesh.bounding_box.vertices[0] + mesh.bounding_box.vertices[5])/2 #
        distances = np.linalg.norm(vertices, axis=1)
        vertices /= (1.03 * np.max(distances))
        # showpoint = np.array(vertices)
        # show_point(showpoint)
    elif cate == 'bottle':
        vertices = mesh.vertices - (
                    mesh.bounding_box.vertices[7] + mesh.bounding_box.vertices[2]) / 2  #
        distances = np.linalg.norm(vertices, axis=1)
        vertices /= (1.03 * np.max(distances))
        # showpoint = np.array(vertices)
        # show_point(showpoint)
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='change seg to original ply')
    parser.add_argument('--category', type=str, default='bottle')
    parser.add_argument('--mode', type=str, default='eval')
    args = parser.parse_args()

    data_root = '../datasets'
    split_path = '../DIF/split'

    obj_path = os.path.join(data_root, 'obj', args.category)
    ply_path = os.path.join('../DIF/eval', args.category)

    expansion_dict = {'bottle': 2, 'knife': 3}
    expansion_number = expansion_dict[args.category]

    with open(os.path.join(split_path, args.mode, args.category + '.txt'), 'r') as f:
        file_name = []
        file_name = f.readlines()
        file_name = [i.rstrip().split('/')[0] for i in file_name]

    for file in file_name:
        for i in range(expansion_number):
            mesh_org = trimesh.load(os.path.join(obj_path, file, '{}.ply'.format(i)))
            mesh_org = scale_to_unit_sphere_dif(mesh_org,args.category)
            vertices_org,_ = trimesh.sample.sample_surface_even(mesh_org, 16000)
            showpoint_org = torch.from_numpy(np.array(vertices_org))
            # show_point(showpoint_org)

            mesh = trimesh.load(os.path.join(ply_path,'{}/{}'.format(file,i),'checkpoints/test.ply'))
            showpoint = torch.from_numpy(np.array(mesh.vertices))
            # show_point(showpoint,showpoint_org)

            dis = (((showpoint_org.unsqueeze(1).expand(-1, showpoint.shape[0], -1) - showpoint.unsqueeze(
                0).expand(showpoint_org.shape[0], -1, -1)) ** 2).sum(-1).sqrt())
            idx = dis.argmin(-1)
            mesh_colors = np.asarray(mesh.visual.vertex_colors[idx,:3])
            mesh_colors = np.clip(mesh_colors / 255, 0, 1)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(showpoint_org)
            pcd.colors = o3d.utility.Vector3dVector(mesh_colors)
            o3d.io.write_point_cloud(os.path.join(obj_path, file, 'seg{}.ply'.format(i)), pcd, True)