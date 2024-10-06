import torch
from torch import nn, Tensor
from typing import Optional


class TransformerRefinement(nn.Module):

    def __init__(self, args, d_model, nhead, dim_feedforward=2048, dropout=0.1, k=20):
        super().__init__()
        self.k = k
        self.pos_embed = nn.Sequential(
            nn.Conv1d(3, d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.pos_embed_cross = nn.Sequential(
            nn.Conv1d(3, d_model+args.num_points, kernel_size=1, bias=False),#d_model+
            nn.BatchNorm1d(d_model+args.num_points),
            nn.LeakyReLU(negative_slope=0.2),
        )
        # self atten用于query之间的自注意力
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.heat_attn = nn.MultiheadAttention(args.num_points, nhead, dropout=dropout, kdim=d_model, vdim=d_model, batch_first=True)
        self.query_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, kdim=args.num_points, vdim=args.num_points, batch_first=True)

        # Implementation of Feedforward model
        self.ffn_query = nn.Sequential(nn.Conv1d(d_model, dim_feedforward, kernel_size=1, bias=False),
                                       nn.LeakyReLU(0.2),
                                       nn.Dropout(dropout),
                                       nn.Conv1d(dim_feedforward, d_model, kernel_size=1, bias=False))
        self.ffn_heat = nn.Sequential(nn.Conv1d(args.num_points, dim_feedforward, kernel_size=1, bias=False),
                                      nn.LeakyReLU(0.2),
                                      nn.Dropout(dropout),
                                      nn.Conv1d(dim_feedforward, args.num_points, kernel_size=1, bias=False))

        self.norm_self = nn.BatchNorm1d(d_model)
        self.norm_cq = nn.BatchNorm1d(d_model)
        self.norm_fq = nn.BatchNorm1d(d_model)
        self.norm_ch = nn.BatchNorm1d(args.query_num)
        self.norm_fh = nn.BatchNorm1d(args.query_num)
        self.dropout_self = nn.Dropout(dropout)
        self.dropout_cq = nn.Dropout(dropout)
        self.dropout_fq = nn.Dropout(dropout)
        self.dropout_ch = nn.Dropout(dropout)
        self.dropout_fh = nn.Dropout(dropout)

    # 生成pos embed
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + self.pos_embed(pos)

    def forward(self, heat_feat, query_feat, query_pos):
        num_points = heat_feat.shape[2]
        v = query_feat.permute(0,2,1)
        pos_embed = self.pos_embed(query_pos.permute(0, 2, 1))
        q = k = v + pos_embed.permute(0, 2, 1)
        z = self.self_attn(query=q, key=k, value=v)[0]
        z = z.permute(0, 2, 1)
        # add & norm
        query_feat = query_feat + self.dropout_self(z)
        query_feat = self.norm_self(query_feat)

        z_query = self.query_attn(query=query_feat.permute(0, 2, 1), key=heat_feat, value=heat_feat)[0]
        z_heat = self.heat_attn(query=heat_feat,key=query_feat.permute(0, 2, 1),value=query_feat.permute(0, 2, 1))[0]

        # add & norm
        query_feat = query_feat + self.dropout_cq(z_query.permute(0, 2, 1))
        query_feat = self.norm_cq(query_feat)
        # FFN
        query_ffn = self.ffn_query(query_feat)
        query_feat = query_feat + self.dropout_fq(query_ffn)
        query_feat = self.norm_fq(query_feat)
        # add & norm
        heat_feat = heat_feat + self.dropout_ch(z_heat)
        heat_feat = self.norm_ch(heat_feat)

        return heat_feat, query_feat




