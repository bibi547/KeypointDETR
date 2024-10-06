import os
import numpy as np
import random
import json
from glob import glob

from scipy.stats import special_ortho_group

from torch.utils.data import Dataset
from utils.data_utils import naive_read_pcd, normalize_pc, geodesic_heatmaps, add_noise

ID2NAMES = {"02691156": "airplane",
            "02808440": "bathtub",
            "02818832": "bed",
            "02876657": "bottle",
            "02954340": "cap",
            "02958343": "car",
            "03001627": "chair",
            "03467517": "guitar",
            "03513137": "helmet",
            "03624134": "knife",
            "03642806": "laptop",
            "03790512": "motorcycle",
            "03797390": "mug",
            "04225987": "skateboard",
            "04379243": "table",
            "04530566": "vessel",}
NAMES2ID = {v: k for k, v in ID2NAMES.items()}


class KPS_Geodesic_Dataset(Dataset):
    def __init__(self, args, split_file: str, train: bool):
        self.args = args
        self.class_id = NAMES2ID[args.class_name]
        self.pcds = []
        self.kp_indices = []
        self.mesh_names = []
        self.nclasses = 2
        self.augmentation = args.augmentation if train else False
        self.normalize_pc = args.normalize_pc
        annots = json.load(open(args.anno_dir))
        annots = [annot for annot in annots if annot['class_id'] == NAMES2ID[args.class_name]]
        keypoints = dict(
            [(annot['model_id'], [kp_info['pcd_info']['point_index'] for kp_info in annot['keypoints']]) for annot in
             annots])

        split_models = open(os.path.join(args.split_root, split_file)).readlines()
        split_models = [m.split('-')[-1].rstrip('\n') for m in split_models]

        for fn in glob(os.path.join(args.pcd_root, NAMES2ID[args.class_name], '*.pcd')):
            model_id = os.path.basename(fn).split('.')[0]
            if model_id not in split_models:
                continue
            self.kp_indices.append(keypoints[model_id])
            self.pcds.append(naive_read_pcd(fn)[0])
            self.mesh_names.append(model_id)

    def __len__(self):
        return len(self.pcds)

    def __getitem__(self, idx):
        pc = self.pcds[idx]
        mesh_name = self.mesh_names[idx]
        kp_idx = self.kp_indices[idx]

        geodesic_file = os.path.join(self.args.pcd_root, self.class_id, mesh_name+'.txt')
        geo_matrix = np.loadtxt(geodesic_file,delimiter=',')
        heats = geodesic_heatmaps(geo_matrix, self.args)

        heats_feature = -1 * np.ones((heats.shape[0], 50))
        heats_feature[:, :heats.shape[1]] = heats

        if self.normalize_pc:
            pc = normalize_pc(pc)

        if self.augmentation:
            pc = add_noise(pc, sigma=0.004, clip=0.01)
            tr = np.random.rand() * 0.02
            rot = special_ortho_group.rvs(3)
            pc = pc @ rot
            pc += np.array([[tr, 0, 0]])
            pc = pc @ rot.T

        pc = pc.T
        heats_feature = heats_feature.T
        return pc.astype(np.float32), heats_feature.astype(np.float32), mesh_name


if __name__ == '__main__':
    class Args(object):
        def __init__(self):
            self.anno_dir = 'F:/dataset/keypointnet/annotations/all.json'
            self.class_name = 'airplane'
            self.split_root = 'F:/dataset/keypointnet/splits'
            self.pcd_root = 'F:/dataset/keypointnet/pcds'
            self.augmentation = True
            self.normalize_pc = True
            self.landmark_std = 0.2

    data = KPS_Geodesic_Dataset(Args(), 'train.txt', True)
    i = 0
    for pc, heat, _ in data:
        print(pc.shape)
        print(heat.shape)
        i += 1
    print(i)



