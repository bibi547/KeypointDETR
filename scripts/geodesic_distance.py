import os
import numpy as np
import json
from sklearn.manifold import Isomap
from utils.data_utils import naive_read_pcd


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


def geo_distance_metrix(points):
    isomap = Isomap(n_components=2, n_neighbors=5, path_method="auto")
    data_2d = isomap.fit_transform(X=points)
    geo_distance_metrix = isomap.dist_matrix_  # 测地距离矩阵，shape=[n_sample,n_sample]
    return geo_distance_metrix


def write_keypoints_geo_distance_metrix(pcd_file, kp_idx, out_file):
    pcd = naive_read_pcd(pcd_file)[0]
    distance_metrix = geo_distance_metrix(pcd)
    distance_metrix = distance_metrix[kp_idx]
    print("shape",pcd_file,"computed")
    np.savetxt(out_file,distance_metrix,fmt='%f',delimiter=',')


if __name__ == '__main__':
    pcd_root = 'F:/dataset/keypointnet/pcds/'
    write_root = 'F:/dataset/keypointnet/pcds/'
    anno_file = 'F:/dataset/keypointnet/annotations/all.json'
    class_name = 'table'

    pcd_root = os.path.join(pcd_root, NAMES2ID[class_name])
    write_root = os.path.join(write_root, NAMES2ID[class_name])

    annots = json.load(open(anno_file))
    annots = [annot for annot in annots if annot['class_id'] == NAMES2ID[class_name]]
    keypoints = dict(
        [(annot['model_id'], [kp_info['pcd_info']['point_index'] for kp_info in annot['keypoints']]) for annot in
         annots])

    pcd_files = os.listdir(pcd_root)
    for f in pcd_files:
        filename = os.path.splitext(f)[0]
        pcd_file = os.path.join(pcd_root, filename + '.pcd')
        write_file = os.path.join(pcd_root, filename + '.txt')
        kp_idx = keypoints[filename]

        write_keypoints_geo_distance_metrix(pcd_file, kp_idx, write_file)