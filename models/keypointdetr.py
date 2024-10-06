import torch
import torch.nn as nn

from .utils import index_points, SharedMLP1d
from .decoder import TransformerRefinement
from .encoder import DGTBackbone


class KeypointDETR(nn.Module):

    def __init__(self, args):
        super(KeypointDETR, self).__init__()

        self.query_num = args.query_num

        self.backbone = DGTBackbone(args)

        self.heat_decoder = nn.Sequential(SharedMLP1d([args.n_edgeconvs_backbone*64 + args.emb_dims, 512, 128, 64], args.norm),
                                          nn.Dropout(args.dropout),
                                          nn.Conv1d(64, args.query_num, kernel_size=1))
        self.query_decoder = SharedMLP1d([args.num_points, 1024, 256, 128], args.norm)

        self.trans_refine = TransformerRefinement(args=args, d_model=128, nhead=8)

        self.prob_out = nn.Sequential(SharedMLP1d([128, 64], args.norm),
                                      nn.Dropout(args.dropout),
                                      nn.Conv1d(64, 2, kernel_size=1),)
        self.heat_out = nn.Sequential(SharedMLP1d([args.query_num, args.query_num], args.norm),
                                      nn.Dropout(args.dropout),
                                      nn.Conv1d(args.query_num, args.query_num, kernel_size=1))

    def forward(self, x):

        xyz = x.contiguous()  # B,3,N
        xyz_t = x.permute(0, 2, 1).contiguous()

        feat = self.backbone(x)

        heat_feat = self.heat_decoder(feat)  # B,M,N
        query_feat = self.query_decoder(heat_feat.permute(0, 2, 1))  # B,128,M

        h_idx = torch.argmax(heat_feat, dim=2)  # B,M
        p_xyz = index_points(xyz_t, h_idx)
        heat_feat, query_feat = self.trans_refine(heat_feat, query_feat, p_xyz)

        prob = self.prob_out(query_feat)  # B,2,M
        prob = prob.permute(0, 2, 1)
        heat = self.heat_out(heat_feat)

        return prob, heat

    def inference(self, x, gt):
        xyz = x.contiguous()  # B,3,N
        xyz_t = x.permute(0, 2, 1).contiguous()

        feat = self.backbone(x)

        heat_feat = self.heat_decoder(feat)  # B,M,N
        query_feat = self.query_decoder(heat_feat.permute(0, 2, 1))  # B,128,M

        h_idx = torch.argmax(heat_feat, dim=2)  # B,M
        p_xyz = index_points(xyz_t, h_idx)
        heat_feat, query_feat = self.trans_refine(heat_feat, query_feat, p_xyz)

        prob = self.prob_out(query_feat)  # B,2,M
        prob = prob.permute(0, 2, 1)
        heat = self.heat_out(heat_feat)

        h_idx = torch.argmax(heat, dim=2)  # B,M
        p_xyz = index_points(xyz_t, h_idx)

        pred_c = torch.argmax(prob, dim=2).squeeze()  # 预测的所有query类别
        indices = torch.nonzero(pred_c == 1, as_tuple=False).squeeze()
        p_xyz = p_xyz[:, indices, :]

        gt = gt.squeeze()
        mask = torch.all(gt.squeeze() != -1, dim=1)
        gt = gt[mask]
        g_idx = torch.argmax(gt, dim=1)  # B,M
        g_xyz = xyz_t.squeeze()[g_idx]

        return p_xyz.squeeze(), g_xyz.squeeze()
