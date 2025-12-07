import torch
from models.vggt_udf import VGGT_Udf
from third_party.DCUDF.udf_models import Weighted_Dist_UDF
from third_party.vggt.vggt.utils.load_fn import load_and_preprocess_images
from utils.utils import extract_mesh
from utils.chamfer_distance import get_cd_loss
import viser
import os
import glob
import numpy as np

import time

def evaluate(images : torch.Tensor, device="cuda"):
    print("loading models")
    model = VGGT_Udf(use_pretune=True).to(device)
    model.eval()

    images.to(device)
    pcd, udf, color_mask = model(images)

    print("dcudf: start")
    start = time.time()
    pred_mesh = extract_mesh(udf, device=device)

    pcd = pcd.cpu().detach().numpy()
    loss = get_cd_loss(pcd, pred_mesh)
    print(f"cd loss: {loss}")
    end = time.time()
    print(f"dcudf: done {end - start}")

    #for coloring purposes
    colors = images.transpose(0, 2, 3, 1)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)

    return pcd, pred_mesh, colors_flat[color_mask]

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("processing images")
    start = time.time()
    image_names = glob.glob(os.path.join("src/images/safeway", "*"))
    images = load_and_preprocess_images(image_names).to(device)
    end = time.time()

    print(f"done processing images: {end - start}")

    print("start infer")
    start = time.time()
    pred_pcd, pred_mesh, pred_colors = evaluate(images=images, device=device)
    end = time.time()

    print(f"done infer: {end - start}")
    server = viser.ViserServer(host='0.0.0.0', port=8080)

    server.scene.add_point_cloud(
        name="pred point cloud",
        points=pred_pcd,
        colors=pred_colors,
        point_size=0.001
    )

    server.scene.add_mesh_trimesh(
        name="pred mesh",
        mesh=pred_mesh
    )

    print("running server")

    server_url = server.request_share_url()

    print(f"viser server url: {server_url}")

    while True:
        pass