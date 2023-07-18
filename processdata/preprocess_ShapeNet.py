import trimesh
import numpy as np
import copy
import plyfile
import open3d as o3d
import os
import sys, argparse
import shutil
sys.path.append('.')

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def change_shapenet_file_to_DIF_tree():

    obj_path = os.path.join(data_root, args.category)
    inst_list = []
    for sub_name in sorted(os.listdir(obj_path)):

        inst_list.append(sub_name)
        if not os.path.exists(os.path.join(obj_path, sub_name,'models')):
            print('Do not have models file')
        des_path = os.path.join(obj_path, sub_name)
        full_path = os.path.join(obj_path, sub_name, 'models', 'model_normalized.obj')
        if not os.path.exists(os.path.join(obj_path,sub_name,'model_normalized.obj')):
            shutil.move(full_path,des_path)

        for sub_name_file in sorted(os.listdir(des_path)):
            if not (sub_name_file.endswith('.obj')):
                path = os.path.join(des_path, sub_name_file)
                shutil.rmtree(path)
        if os.path.exists(os.path.join(des_path, 'model_normalized.obj')):
            oldname = os.path.join(des_path,'model_normalized.obj')
            newname = os.path.join(des_path,'model.obj')
            os.rename(oldname, newname)

def show_point(points):
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([axis, point_cloud])

def deform_3D(pts,scale_xOz,scale_y):
    '''
    Args:
        pts: [N,3] array, vertices of mesh
        scale_xOz: float, scale size of top face of 3D bbox
        scale_y: float, scale size along Y axis
    Return:
        pts_deformed: [N,3]
    '''

    pts_deformed = pts.copy()
    y_max, y_min = np.amax(pts[:,1]), np.amin(pts[:,1])
    deform_func_xOz = lambda x : ((scale_xOz - 1) / (y_max - y_min) * x + (y_max - scale_xOz * y_min) / (y_max - y_min))
    pts_deformed[:,(0,2)] = pts_deformed[:,(0,2)] * deform_func_xOz(pts_deformed[:,1])[:, np.newaxis]
    pts_deformed[:,1] = pts_deformed[:,1] * scale_y
    return pts_deformed

def norm_pts(pts, norm_size=0.5):
    '''
    Args:
        pts: [N,3]
        norm_size: the sphere's radius of 3D bbox of normed pts
    Return:
        pts_normed: [N,3]
    '''
    radius_bbox = np.amax(abs(pts), axis=0)
    pts_normed = pts / np.linalg.norm(radius_bbox) * norm_size
    return pts_normed

def save_ply(verts, faces, path_ply):
    '''
    Write a ply file to specified path.
    Args:
        verts: [V,3]
        faces: [F,3]
        path_ply: path where ply file be writed.
    '''
    num_verts = verts.shape[0]
    num_faces = faces.shape[0]
    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    for i in range(0, num_verts):
        verts_tuple[i] = tuple(verts[i, :])
    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])
    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces],text=True)
    ply_data.write(path_ply)

def getfiles(dirPath, fileType):
    fileList = []
    for root, dirs, files in os.walk(dirPath):
        for f in files:
            w = [i in f for i in fileType]
            if all(w):
                fileList.append(os.path.join(root, f))
            # else:
            #     print('Not a need file:', f)
    return fileList

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='data preprocess')
    parser.add_argument('--category', type=str, default='bottle')
    parser.add_argument('--expansion', action='store_false', default=True)
    args = parser.parse_args()
    data_root = '../datasets/obj'

    save_path = os.path.join(data_root, args.category)
    cond_mkdir(save_path)
    obj_path = os.path.abspath(save_path)

    objslist = getfiles(obj_path, ['model_normalized.obj'])
    if objslist:
        change_shapenet_file_to_DIF_tree()
    print('change file success !')


    for inst_name in os.listdir(obj_path):
        path_mesh_source = os.path.join(obj_path, inst_name, 'model.obj')
        mesh_source = trimesh.load(path_mesh_source, file_type='obj')
        cond_mkdir(os.path.join(obj_path, inst_name))
        if isinstance(mesh_source, trimesh.Scene):
            geo_list = []
            for k, v in mesh_source.geometry.items():
                geo_list.append(v)
            mesh_source = trimesh.util.concatenate(geo_list)
        mesh_target = copy.deepcopy(mesh_source)

        mesh_target.vertices = deform_3D(mesh_source.vertices, 1, 1)
        save_ply(mesh_target.vertices, mesh_target.faces, os.path.join(obj_path, inst_name, '0.ply'))
        # showpoint = np.array(mesh_target.vertices)
        # show_point(showpoint)

        # Expansion data
        if args.expansion:
            if args.category == 'bottle': deform_list = [(1.1, 1)]
            if args.category ==  'knife': deform_list = [(0.8, 1), (1.3, 1)]
            for i, (scale_xOz, scale_y) in enumerate(deform_list):
                print(scale_xOz, scale_y)
                mesh_target.vertices = deform_3D(mesh_source.vertices, scale_xOz, scale_y)
                path_mesh_target = os.path.join(obj_path, inst_name, '{0}.ply'.format(i+1))
                save_ply(mesh_target.vertices, mesh_target.faces, path_mesh_target)
                # showpoint = np.array(mesh_target.vertices)
                # show_point(showpoint)

    # change file format using meshlab
    fType = ['.ply']
    res = getfiles(obj_path, fType)
    print('load result：')
    os.chdir('/usr/bin') # CD to the folder where meshlabserver.exe is located

    for f in res:
        print(f)
        ipath = f
        opath = f[0:-3] + "obj"
        os.system('meshlabserver -i ' + ipath + ' -o ' + opath + ' -m vn')
        os.system('exit()')
