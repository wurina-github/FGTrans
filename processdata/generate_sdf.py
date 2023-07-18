import numpy as np
from mesh_to_sdf import get_surface_point_cloud,sample_sdf_near_surface
import trimesh
import pyrender
import scipy.io as sio
import os
import open3d as o3d
import sys,argparse
sys.path.append('./')

def show_point(points):
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5,origin=[0,0,0])
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.paint_uniform_color([1,0.706,0])
    o3d.visualization.draw_geometries([axis, point_cloud])

def scale_to_unit_sphere_dif(mesh,cate):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()
    elif cate == 'knife':
        vertices = mesh.vertices - (mesh.bounding_box.vertices[0] + mesh.bounding_box.vertices[5])/2
        distances = np.linalg.norm(vertices, axis=1)
        vertices /= (1.03 * np.max(distances))
        # showpoint = np.array(vertices)
        # show_point(showpoint)
    elif cate == 'bottle':
        vertices = mesh.vertices - (
                    mesh.bounding_box.vertices[7] + mesh.bounding_box.vertices[2]) / 2
        distances = np.linalg.norm(vertices, axis=1)
        vertices /= (1.03 * np.max(distances))
        # showpoint = np.array(vertices)
        # show_point(showpoint)
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

def sample_sdf(file_path):
    mesh = trimesh.load(file_path)
    mesh = scale_to_unit_sphere_dif(mesh,args.category)

    surface = get_surface_point_cloud(mesh, surface_point_method='scan', scan_count=100, sample_point_count=500000,
                                    calculate_normals=True)
    surface_p = surface.points
    surface_n = surface.normals
    surface_data = np.concatenate((surface_p, surface_n), axis=1)

    free_p, free_sdf = sample_sdf_near_surface(mesh, surface_point_method='scan', sign_method='depth',
                                               sample_point_count=500000)
    free_sdf = free_sdf.reshape(-1, 1)
    free_data = np.concatenate((free_p, free_sdf), axis=1)
    return free_data, surface_data

def show_sdf(points_free, sdf_free, points_surface=None):

    colors = np.zeros(points_free.shape)
    colors[sdf_free < 0, 2] = 1
    colors[sdf_free > 0, 0] = 1

    scene = pyrender.Scene()
    cloud_free = pyrender.Mesh.from_points(points_free, colors=colors)
    scene.add(cloud_free)

    if points_surface is not None:
        colors = np.zeros(points_surface.shape)
        colors[:, 1] = 1
        cloud_surface = pyrender.Mesh.from_points(points_surface, colors=colors)
        scene.add(cloud_surface)

    pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)

def write_mat_data(save_path, file, points_with_sdf, points_with_normal,idx):

    pts_sdf  = points_with_sdf
    pts_normal = points_with_normal
    if not os.path.exists(os.path.join(save_path, 'free_space_pts',file)):
        os.makedirs(os.path.join(save_path, 'free_space_pts',file))
    if not os.path.exists(os.path.join(save_path, 'surface_pts_n_normal',file)):
        os.makedirs(os.path.join(save_path, 'surface_pts_n_normal',file))
    free_points_path = os.path.join(save_path, 'free_space_pts', file, '{}'.format(idx) +'.mat')
    surface_points_path = os.path.join(save_path, 'surface_pts_n_normal', file, '{}'.format(idx) + '.mat')
    sio.savemat(free_points_path, {'p_sdf': pts_sdf})
    sio.savemat(surface_points_path, {'p': pts_normal})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate sdfs')
    parser.add_argument('--category', type=str, default='bottle')
    parser.add_argument('--mode', type=str, default='eval')
    parser.add_argument('--vis', action= 'store_true')
    args = parser.parse_args()
    data_root = '../datasets'
    split_path = '../DIF/split'

    save_path = os.path.join(data_root, args.category)
    obj_path = os.path.join(data_root, 'obj', args.category)
    expansion_dict = {'bottle':2, 'knife':3}
    expansion_number=expansion_dict[args.category]

    if not os.path.exists(os.path.join(save_path, 'free_space_pts')):
        os.makedirs(os.path.join(save_path, 'free_space_pts'))
    if not os.path.exists(os.path.join(save_path, 'surface_pts_n_normal')):
        os.makedirs(os.path.join(save_path, 'surface_pts_n_normal'))

    with open(os.path.join(split_path, args.mode, args.category + '.txt'), 'r') as f:
        file_name = []
        file_name = f.readlines()
        file_name = [i.rstrip().split('/')[0] for i in file_name]
    for i,file in enumerate(os.listdir(obj_path)):
        for file in file_name:
            for i in range(expansion_number):
                if os.path.exists(os.path.join(save_path, 'free_space_pts', file, '{}'.format(i) +'.mat')):
                    continue
                points_with_sdf, points_with_normal = sample_sdf(os.path.join(obj_path, file, '{}.obj'.format(i)))
                # show_sdf(points_with_sdf[0], points_with_sdf[1], points_with_normal.points)
                write_mat_data(save_path, file, points_with_sdf, points_with_normal, i)
                print('success,{}'.format(os.path.join(save_path, 'free_space_pts', file, '{}'.format(i) +'.mat')))
                if args.vis:
                    # free space pts
                    pts_free = sio.loadmat(os.path.join(save_path, 'free_space_pts', file, '{}'.format(i) +'.mat'))['p_sdf']
                    # surface pts n normal
                    pts_normal = sio.loadmat(os.path.join(save_path, 'surface_pts_n_normal', file, '{}'.format(i) +'.mat'))['p']
                    show_sdf(pts_free[:, :3], pts_free[:, 3])
                    show_sdf(pts_free[:, :3], pts_free[:, 3], pts_normal[:, :3])

