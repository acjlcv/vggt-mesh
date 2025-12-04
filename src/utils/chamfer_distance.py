import trimesh
import numpy as np
from scipy.spatial import cKDTree

def get_cd_loss_mesh(mesh_a : trimesh.Trimesh, mesh_b : trimesh.Trimesh, num_samples=30000):
    pcd_a = trimesh.sample.sample_surface(mesh_a, num_samples)[0]
    pcd_b = trimesh.sample.sample_surface(mesh_b, num_samples)[0]

    a_ktree = cKDTree(pcd_a)
    a_dist, a_idx = a_ktree.query(pcd_b)
    b_to_a_cd = np.mean(np.square(a_dist))

    b_ktree = cKDTree(pcd_b)
    b_dist, b_idx = b_ktree.query(pcd_a)
    a_to_b_cd = np.mean(np.square(b_dist))

    return b_to_a_cd + a_to_b_cd

def get_cd_loss(gt_pcd, mesh_a : trimesh.Trimesh, num_samples=30000):
    pcd_a = trimesh.sample.sample_surface(mesh_a, num_samples)[0]

    a_ktree = cKDTree(pcd_a)
    a_dist, a_idx = a_ktree.query(gt_pcd)
    b_to_a_cd = np.mean(np.square(a_dist))

    b_ktree = cKDTree(gt_pcd)
    b_dist, b_idx = b_ktree.query(pcd_a)
    a_to_b_cd = np.mean(np.square(b_dist))

    return b_to_a_cd + a_to_b_cd
