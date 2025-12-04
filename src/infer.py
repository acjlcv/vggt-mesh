import torch
from models.vggt_udf import VGGT_Udf
from third_party.DCUDF.udf_models import Weighted_Dist_UDF
from third_party.vggt.vggt.utils.load_fn import load_and_preprocess_images
from utils.utils import extract_mesh
from utils.chamfer_distance import get_cd_loss
import viser
import os
import glob

import time

def evaluate(images : torch.Tensor, device="cuda"):
    print("loading models")
    model = VGGT_Udf(use_pretune=True).to(device)
    model.eval()


    images.to(device)
    pcd, udf = model(images)

    print("dcudf: start")
    start = time.time()
    pred_mesh = extract_mesh(udf, device=device)
    loss = get_cd_loss(pcd, pred_mesh, device=device)
    print(f"cd loss: {loss}")
    end = time.time()
    print(f"dcudf: done {end - start}")

    return pcd, pred_mesh

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
    pred_pcd, pred_mesh = evaluate(images=images, device=device)
    end = time.time()

    print(f"done infer: {end - start}")
    server = viser.ViserServer(host='0.0.0.0', port=8080)

    server.scene.add_point_cloud(
        name="pred point cloud",
        points=pred_pcd,
        point_size=0.001
    )

    server.scene.add_mesh_trimesh(
        name="pred mesh",
        mesh=pred_mesh
    )

    print("running server")
    while True:
        pass