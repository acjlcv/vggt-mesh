import torch
from torch import nn
from models.layers.losf import LoSF
from utils.utils import *
import sys

import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from third_party.vggt.vggt.models.vggt import VGGT

#wraper instaed of creating a new head
class VGGT_Udf(nn.Module):
    def __init__(self, use_pretune=True, device="cpu"):
        super().__init__()

        #use vggt for patching information + pointcloud/camera estimation
        self.vggt = VGGT(enable_camera=True, enable_point=True, enable_depth=True, enable_track=True).to(device)

        #use losf to convert dpt_head information into udf
        self.losf = LoSF().to(device)

        if use_pretune:
            print("loading vggt")
            _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
            self.vggt.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

            print("loading losf")
            losf_state_dict = torch.load("third_party/losf/pretrained/train-total-uniform.ckpt", map_location=device)
            self.losf.load_state_dict(losf_state_dict["state_dict"])

    #note: dcudf mesh extraction can be done outside of the mesh
    def forward(self, images : torch.Tensor, conf_threshold=0.75, resolution=256, radius=0.018):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        """

        #vggt
        print("vggt: infer")
        start = time.time()
        predictions = self.vggt(images)
        end = time.time()
        print(f"vggt: infer done {end - start}")

        print("vggt: post proc")
        start = time.time()
        #vggt post processing for pcd and conf
        pcd = predictions["world_points"].reshape(-1, 3)
        conf = predictions["world_points_conf"].reshape(-1)

        center = torch.mean(pcd, dim=0)
        pcd_centered = pcd - center

        conf_thresh_val = torch.quantile(conf, conf_threshold)
        conf_mask = (conf >= conf_thresh_val) & (conf > 0.1)

        proc_pcd = pcd_centered[conf_mask]

        end = time.time()
        print(f"vggt: post proc done {end - start}")

        print("losf: pre proc")
        start = time.time()

        verts, _, _ = normalize_pcd(proc_pcd.detach().numpy())

        #losf preprocessing for patches and queries
        patches = extract_patches(verts, radius=radius, resolution=resolution)
        patch_verts = torch.from_numpy(patches["PatchVerts"]).float()
        queries = torch.from_numpy(patches["Queries"]).float()
        scale_factors = patches["ScaleFactors"]
        queries_idx = patches["Queries_IDX"]

        queries = queries.unsqueeze(1)
        vecs_q = patch_verts - queries

        num_queries = queries.shape[0]
        losf_batch_size = min(2048 * 1, num_queries)
        losf_num_batches = num_queries // losf_batch_size
        losf_num_batches += 0 if num_queries % losf_num_batches else 1

        end = time.time()
        print(f"losf: pre proc done {end - start}")

        print("losf: iterative batched infer")
        start = time.time()

        #losf udf predictions
        pred_udf = np.zeros(num_queries, dtype=np.float32)
        for i in range(losf_num_batches):
            start = i * losf_batch_size
            end = np.min([start + losf_batch_size, num_queries])
            pred_udf_batch, displacements = self.losf(
                patch_verts[start:end],
                vecs_q[start:end],
                queries[start:end]
            )
            pred_udf_batch = pred_udf_batch.detach().numpy()
            pred_udf[start:end] = pred_udf_batch[:, 0]

        end = time.time()
        print(f"losf: iterative batched infer done {end - start}")

        #losf udf postprocessing
        print("losf: post proc")
        start = time.time()

        grids = get_grid_coords(resolution)
        num_grids = grids.shape[0]

        udf = np.zeros(num_grids) + 10
        pred_udf = pred_udf / scale_factors
        udf[queries_idx] = pred_udf

        udf = torch.from_numpy(udf).float()

        end = time.time()
        print(f"losf: post proc done {end - start}")

        return proc_pcd, udf
