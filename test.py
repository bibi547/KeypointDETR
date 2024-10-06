import tqdm
import click
import torch
import numpy as np

from scipy.spatial.distance import cdist
from pl_model import LitModel
from data.st_data import KPS_Geodesic_Dataset, NAMES2ID
from utils.metrics import get_cd, hungary_iou

@click.command()
@click.option('--checkpoint', type=str, default='E:/code/keypoint/Saliency_git/runs/keypoint_saliency/version_3/checkpoints/last.ckpt')
@click.option('--gpus', default=1)
def run(checkpoint, gpus):
    model = LitModel.load_from_checkpoint(checkpoint).cuda()
    model.eval()

    args = model.hparams.args
    test_file = 'F:/dataset/keypointnet/splits/test.txt'
    mesh_root = 'F:/dataset/keypointnet/ShapeNetCore.v2.ply'
    class_id = NAMES2ID[args.class_name]
    dataset = KPS_Geodesic_Dataset(args, test_file, False)

    mcd = []
    hmiou = {}

    for i in range(11):
        key = i * 0.01
        hmiou[key] = []

    for i in tqdm.tqdm(range(len(dataset))):
        pc, heat, mesh_name = dataset[i]
        # mesh
        # print(mesh_name)
        # mesh = trimesh.load(os.path.join(mesh_root, class_id, mesh_name + '.ply'))

        pc = torch.tensor(pc, dtype=torch.float32).unsqueeze(0).cuda()
        heat = torch.tensor(heat, dtype=torch.float32).unsqueeze(0).cuda()
        # pred
        with torch.no_grad():
            pts, gts = model.infer(pc, heat)
            pts = pts.cpu().numpy()
            gts = gts.cpu().numpy()

        dists = cdist(gts, pts, metric='euclidean')
        cd = get_cd(dists)
        mcd.append(cd)

        for i in range(11):
            key = i * 0.01
            hiou = hungary_iou(dists, key)
            hmiou[key].append(hiou)

        # visualization result
        # gt_pts = [trimesh.primitives.Sphere(radius=0.02, center=pt).to_mesh() for pt in gts]
        # for pt in gt_pts:
        #     pt.visual.vertex_colors = (255, 0, 0, 255)
        # pred_pts = [trimesh.primitives.Sphere(radius=0.02, center=pt).to_mesh() for pt in pts]
        # for pt in pred_pts:
        #     pt.visual.vertex_colors = (0, 255, 0, 255)
        # trimesh.Scene([mesh] + gt_pts + pred_pts).show()

    for i in range(11):
        key = i * 0.01
        print(np.mean(hmiou[key]))
    print(np.mean(mcd))


if __name__ == "__main__":
    run()
