import torch
from torch import nn
from third_party.vggt.vggt.models.vggt import VGGT

class VGGT_Mesh(nn.Module):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        #use vggt for patching information + pointcloud/camera estimation
        self.vggt = VGGT(enable_camera=True, enable_point=True, enable_depth=False, enable_track=False)

        #use losf to convert dpt_head information into udf/mesh

    def forward(self, x):