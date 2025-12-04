import torch
from torch import nn
import numpy as np
import os
from joblib import Parallel, delayed
from torch_geometric.nn import fps
from scipy.spatial import cKDTree

from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from third_party.DCUDF.udf_models import Weighted_Dist_UDF
from third_party.DCUDF.mesh_extraction_on_UDF import Dcudf_on_UDF

#losf utils
def get_grid_coords(resolution: int = 128):
    x = torch.linspace(-0.52, 0.52, resolution)
    y = torch.linspace(-0.52, 0.52, resolution)
    z = torch.linspace(-0.52, 0.52, resolution)
    grid = torch.meshgrid(x, y, z)
    coords = torch.stack(grid, dim=-1)
    coords = coords.reshape(-1, 3)
    return coords

def ball_query(pcd, queries, radius):
    # construct a KDTree
    tree = cKDTree(pcd)
    # indices = [tree.query_ball_point(query, r=radius) for query in queries]
    indices = tree.query_ball_point(queries, r=radius)
    return indices

def process_single_patch(i, filtered_query_idx, filtered_indices, coords, verts):
    query_idx = filtered_query_idx[i]
    idx = filtered_indices[i]
    query_0 = coords[query_idx]
    verts_sel = verts[idx]
    # normalize to a sphere with radius 1
    verts_query = np.vstack([query_0, verts_sel])
    translation = np.mean(verts_query, axis=0)
    verts_query -= translation
    max_dist = np.max(np.linalg.norm(verts_query, axis=1))
    scale_factor = 1.0 / max_dist
    verts_query = verts_query * scale_factor
    query = verts_query[0]
    verts_sel = verts_query[1:]
    if len(idx) == 128:
        verts_sel = verts_sel
    elif len(idx) > 128:
        ratio = 128 / len(idx)
        sel_idx = fps(torch.tensor(verts_sel), ratio=ratio)
        verts_sel = verts_sel[sel_idx]
    else:
        mean_v = np.mean(verts_sel, axis=0)
        padding_length = 128 - len(idx)
        verts_sel = np.vstack([verts_sel, np.tile(mean_v, (padding_length, 1))])
    return verts_sel, query, scale_factor

def normalize_pcd(pcd, bounds=0.5):
    #normalizing vertices to be [-bounds, bounds] from losf
    min_v = np.min(pcd, axis=0)
    max_v = np.max(pcd, axis=0)
    scale = 1 / np.max(np.abs(max_v - min_v))
    bias = (- bounds * max_v - bounds * min_v) / (max_v - min_v)
    pcd = pcd * scale + bias

    return pcd, scale, bias

def scale_pcd(pcd, scale, bias):
    pcd = pcd * scale + bias
    return pcd


def extract_patches(verts: torch.Tensor, radius: float=0.018, resolution: int=256):
    """
    radius r taken from the losf paper
    """
    coords = get_grid_coords(resolution)
    indices = ball_query(verts, coords, radius)

    # filter index with less than 5 points
    pts_in_lengths = list(map(len, indices))
    pts_in_lengths = np.array(pts_in_lengths)

    print(np.min(pts_in_lengths), np.max(pts_in_lengths), len(pts_in_lengths))

    filtered_query_idx = np.where((2000 > pts_in_lengths) & (pts_in_lengths > 1000))[0]
    filtered_indices = indices[filtered_query_idx]

    print(len(filtered_indices))

    # multi-threading to process patches
    results = Parallel(n_jobs=os.cpu_count() - 2)(
        delayed(process_single_patch)(i, filtered_query_idx, filtered_indices, coords, verts) for i in
        tqdm(range(len(filtered_indices))))
    PatchVerts = np.zeros((len(filtered_indices), 128, 3))
    Queries = np.zeros((len(filtered_indices), 3))
    ScaleFactors = np.zeros(len(filtered_indices))

    for i, (verts_sel, query, scale_factor) in enumerate(results):
        PatchVerts[i] = verts_sel
        Queries[i] = query
        ScaleFactors[i] = scale_factor

    patches = {"PatchVerts": PatchVerts.astype(np.float32), "Queries": Queries.astype(np.float32),
             "ScaleFactors": ScaleFactors.astype(np.float32), "Queries_IDX": filtered_query_idx}

    return patches

def query_function(input_pcd, query_points, device):
    udf_model = Weighted_Dist_UDF()
    # udf_model = Weighted_BNN_UDF(K=40)

    # device_id = device[-1]
    # udf_model = nn.DataParallel(udf_model, device_ids=[int(device_id)])

    udf_model = udf_model.to(device)
    state = torch.load(
            os.path.join(
                "third_party/DCUDF/pretrained_models/UDF/gf_sf_250", "udf_model_best.t7"
            ),
            map_location=device,
        )

    udf_model.load_state_dict(
        state
    )

    udf_model.eval()
    return udf_model.forward(input_pcd, query_points)

#returns trimesh
def extract_mesh(udf, mesh_resolution=256, threshold=0.003, is_cut=True, laplacian_weight=4000, device="cuda"):
    extractor = Dcudf_on_UDF(
            query_function=query_function,
            udf_field=udf,
            max_iter=300,
            resolution=mesh_resolution,
            threshold=threshold,
            is_cut=is_cut,
            bound_min=torch.tensor([-0.52, -0.52, -0.52]),
            bound_max=torch.tensor([0.52, 0.52, 0.52]),
            input_pcd=None,
            laplacian_weight=laplacian_weight,
            device=device,
        )

    mesh = extractor.optimize()
    return mesh