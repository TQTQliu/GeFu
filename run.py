from lib.config import cfg, args
from lib.utils.ply_utils import *
import numpy as np
import os
import glob

def run_dataset():
    from lib.datasets import make_data_loader
    import tqdm

    cfg.train.num_workers = 0
    data_loader = make_data_loader(cfg, is_train=False)
    for batch in tqdm.tqdm(data_loader):
        pass

def run_network():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    from lib.utils.data_utils import to_cuda
    import tqdm
    import torch
    import time

    network = make_network(cfg).cuda()
    load_network(network, cfg.trained_model_dir, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    total_time = 0
    for batch in tqdm.tqdm(data_loader):
        batch = to_cuda(batch)
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            network(batch)
            torch.cuda.synchronize()
            total_time += time.time() - start
    print(total_time / len(data_loader))

def run_evaluate():
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils import net_utils
    import time

    network = make_network(cfg).cuda()
    net_utils.load_network(network,
                           cfg.trained_model_dir,
                           resume=cfg.resume,
                           epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    net_time = []
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                if k == 'rendering_video_meta':
                    for i in range(len(batch[k])):
                        for v in batch[k][i]:
                            batch[k][i][v] = batch[k][i][v].cuda()
                else:
                    batch[k] = batch[k].cuda()
        if cfg.save_video:
            with torch.no_grad():
                network(batch)
        else:
            with torch.no_grad():
                torch.cuda.synchronize()
                start_time = time.time()
                output = network(batch)
                torch.cuda.synchronize()
                end_time = time.time()
            net_time.append(end_time - start_time)
            evaluator.evaluate(output, batch)
            
    if not cfg.save_video:
        evaluator.summarize()
        if len(net_time) > 1:
            # print('net_time: ', np.mean(net_time[1:]))
            print('FPS: ', 1./np.mean(net_time[1:]))
        else:
            # print('net_time: ', np.mean(net_time))
            print('FPS: ', 1./np.mean(net_time))

    
    if cfg.save_ply:
        dataset_name = cfg.train_dataset_module.split('.')[-2]
        ply_dir = os.path.join(cfg.result_dir, 'pointclouds', dataset_name)
        for item in os.listdir(ply_dir):
            data_dir = os.path.join(ply_dir, item)
            img_dir = os.path.join(data_dir, 'images')
            depth_dir = os.path.join(data_dir, 'depth')
            cam_dir = os.path.join(data_dir, 'cam')
            img_ls = glob.glob(os.path.join(img_dir, '*.png'))
            img_name = [os.path.basename(im).split('.')[0] for im in img_ls]

            # for the final point cloud
            vertexs = []
            vertex_colors = []
            
            for name in img_name:
                ref_name = name
                
                ref_intrinsics, ref_extrinsics = read_camera_parameters(os.path.join(cam_dir, ref_name+'.txt'))
                ref_img = read_img(os.path.join(img_dir, ref_name+'.png'))
                ref_depth_est = read_pfm(os.path.join(depth_dir, ref_name+'.pfm'))[0]

                height, width = ref_depth_est.shape[:2]
                x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
                x, y = x.reshape(-1), y.reshape(-1)
                depth = ref_depth_est.reshape(-1)
                color = ref_img.reshape(-1,3)
                xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                                    np.vstack((x, y, np.ones_like(x))) * depth)
                xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                                    np.vstack((xyz_ref, np.ones_like(x))))[:3]
                vertexs.append(xyz_world.transpose((1, 0)))
                vertex_colors.append((color * 255).astype(np.uint8))
            vertexs = np.concatenate(vertexs, axis=0)
            vertex_colors = np.concatenate(vertex_colors, axis=0)
            scene = os.path.basename(data_dir)
            ply_path = os.path.join(data_dir, f'{scene}.ply')
            print(f'saving {ply_path}')
            storePly(ply_path, vertexs, vertex_colors)
            
            ## point cloud --> mesh
            import open3d as o3d
            import trimesh
            pcd = o3d.io.read_point_cloud(ply_path)
            pcd.estimate_normals()
            
            # estimate radius for rolling ball
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = 1.5 * avg_dist   

            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd,
                    o3d.utility.DoubleVector([radius, radius * 2]))
            
            # create the triangular mesh with the vertices and faces from open3d
            tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                                    vertex_normals=np.asarray(mesh.vertex_normals))
            
            trimesh.convex.is_convex(tri_mesh)
            ply_path = os.path.join(data_dir, f'{scene}_mesh.ply')
            trimesh.exchange.export.export_mesh(tri_mesh, ply_path)
            

                
if __name__ == '__main__':
    globals()['run_' + args.type]()
