import torch
import torch.nn as nn
from typing import Optional


class SharedMLP1d(nn.Module):
    def __init__(self, channels, norm):
        super(SharedMLP1d, self).__init__()

        norm_layer = get_norm_layer_1d(norm)

        self.conv = nn.Sequential(*[
            nn.Sequential(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=False),
                          norm_layer(channels[i]),
                          nn.LeakyReLU(0.2))
            for i in range(1, len(channels))
        ])

    def forward(self, x):
        return self.conv(x)


def get_norm_layer_1d(norm):
    if norm == 'instance':
        norm_layer = nn.InstanceNorm1d
    elif norm == 'batch':
        norm_layer = nn.BatchNorm1d
    else:
        assert 0, "not implemented"
    return norm_layer


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def knn(x, k: int):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k: int, idx: Optional[torch.Tensor] = None):
    batch_size = x.size(0)
    n_channels = x.size(1)      # fixed
    num_points = x.size(2)      # dynamic
    x = x.view(batch_size, n_channels, -1)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx_base = idx_base.long()
    idx = idx + idx_base
    idx = idx.contiguous().view(-1)

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    feature = x.view(-1, n_channels)[idx, :]
    feature = feature.view(batch_size, -1, k, n_channels)
    x = x.view(batch_size, -1, 1, n_channels).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature













