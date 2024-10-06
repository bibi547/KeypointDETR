import numpy as np
import trimesh
import os


def geodesic_heatmaps(geodesic_matrix, args):
    ys = []
    for i, dist in enumerate(geodesic_matrix):
        y = np.exp(- dist ** 2 / (2 * args.landmark_std ** 2))
        ys.append(y)
    return np.asarray(ys).T


def naive_read_pcd(path):
    lines = open(path, 'r').readlines()
    idx = -1
    for i, line in enumerate(lines):
        if line.startswith('DATA ascii'):
            idx = i + 1
            break
    lines = lines[idx:]
    lines = [line.rstrip().split(' ') for line in lines]
    data = np.asarray(lines)
    pc = np.array(data[:, :3], dtype=np.float64)
    colors = np.array(data[:, -1], dtype=np.int32)
    colors = np.stack([(colors >> 16) & 255, (colors >> 8) & 255, colors & 255], -1)
    return pc, colors


def add_noise(x, sigma=0.015, clip=0.05):
    noise = np.clip(sigma*np.random.randn(*x.shape), -1*clip, clip)
    return x + noise


def normalize_pc(pc):
    pc = pc - pc.mean(0)
    pc /= np.max(np.linalg.norm(pc, axis=-1))
    return pc

