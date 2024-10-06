import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .utils import SharedMLP1d, index_points, knn


class DGTBlock(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Conv1d(d_points, d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.fc2 = nn.Sequential(
            nn.Conv1d(d_model, d_points, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_points),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.fc_delta = nn.Sequential(
            nn.Conv2d(3, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.fc_gamma = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.w_qs = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.w_ks = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.w_vs = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)

        self.k = k

    def forward(self, feats, pos):

        d_idx = knn(feats, self.k)  # dynamic k idx
        knn_idx = d_idx
        knn_pos = index_points(pos.permute(0,2,1), knn_idx)

        pre = feats
        x = self.fc1(feats)
        q = self.w_qs(x)  # b x f x n
        k = index_points(self.w_ks(x).permute(0,2,1), knn_idx).permute(0, 3, 2, 1)  # b x f x k x n
        v = index_points(self.w_vs(x).permute(0,2,1), knn_idx).permute(0, 3, 2, 1)  # b x f x k x n

        pos = pos.permute(0, 2, 1)[:, :, None] - knn_pos
        pos = pos.permute(0, 3, 2, 1)
        pos_enc = self.fc_delta(pos)  # b x f x k x n

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x f x k x n

        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)  # b x f x n
        res = self.fc2(res) + pre  # b x f x n

        return res


class DGTBackbone(nn.Module):
    def __init__(self, args):
        super(DGTBackbone, self).__init__()

        self.k = args.k
        self.dynamic = args.dynamic

        self.smlp = SharedMLP1d([3, 64], args.norm)
        self.encoders = nn.ModuleList()
        for _ in range(args.n_edgeconvs_backbone):
            self.encoders.append(DGTBlock(d_points=64, d_model=128, k=args.k))

        self.smlp_1 = SharedMLP1d([args.n_edgeconvs_backbone*64, args.emb_dims], args.norm)

        # global pooling
        if args.global_pool_backbone == 'avg':
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif args.global_pool_backbone == 'max':
            self.pool = nn.AdaptiveMaxPool1d(1)
        else:
            assert 0

    def forward(self, x):
        xyz = x
        x = self.smlp(x)
        xs = []
        for conv in self.encoders:
            x = conv(x, xyz)
            xs.append(x)
        x = torch.cat(xs, dim=1)

        x = self.smlp_1(x)
        x_pool = self.pool(x)
        x_pool = x_pool.expand(x_pool.shape[0], x_pool.shape[1], x.shape[2])

        x = torch.cat((x_pool, torch.cat(xs, dim=1)), dim=1)

        return x




