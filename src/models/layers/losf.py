import torch

import torch.nn as nn
from models.layers.feature_attn_layer import FeatureAttnLayer
from models.layers.pointnet import ResnetPointnet

class LoSF(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = FeatureAttnLayer(pts_dim=64, vec_dim=64, out_dim=128, num_self_attn=4)
        self.pts_net = ResnetPointnet(c_dim=64, dim=3, hidden_dim=64)
        self.vec_net = ResnetPointnet(c_dim=64, dim=3, hidden_dim=64)
        self.pts_denoise_net = ResnetPointnet(c_dim=64, dim=3, hidden_dim=64)

    #input_data = vertices, vectors from query q, and query point q
    def forward(self, pts: torch.Tensor, vecs_q: torch.Tensor, query: torch.Tensor):
        query = query.unsqueeze(1)
        query = torch.tile(query, (1, pts.shape[1], 1))

        pts_feature = self.pts_net(pts)
        pts_denoise_feature = self.pts_denoise_net(pts)
        vecs_feature = self.vec_net(vecs_q)
        distance = torch.norm(vecs_q, dim=2)
        # pred_udf, displacement = self.net(pts_feature, vecs_feature, distance)
        pred_udf, displacement = self.net(
            pts_feature, vecs_feature, pts_denoise_feature, distance
        )

        return pred_udf, displacement