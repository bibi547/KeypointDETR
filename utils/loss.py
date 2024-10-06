import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from .matcher import HungarianMatcher


class Criterion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.matcher = HungarianMatcher(cost_prob=100, cost_heat=1)
        self.loss_h = nn.MSELoss()
        self.loss_c = FocalLoss(num_classes=2)

    def forward(self, probs, heats, g_heat):
        indices = self.matcher(probs, heats, g_heat)

        loss = 0
        for i, idx in enumerate(indices):  # for batch_size
            gheat = g_heat[i, :, :].squeeze()
            mask = torch.all(gheat != -1, dim=1)
            gheat = gheat[mask]
            gheat = gheat[idx[1], :]

            gprob = torch.zeros(probs.shape[1], dtype=torch.long).to(probs.device)
            gprob[idx[0]] = 1

            pheat = heats[i, idx[0], :]
            prob = probs[i]
            loss_h = self.loss_h(pheat.squeeze(), gheat)
            loss_c = self.loss_c(prob.squeeze(), gprob)
            loss = loss + loss_c + loss_h * 2

        return loss


class FocalLoss(torch.nn.Module):
    def __init__(self, num_classes, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(num_classes, 1))
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes

    def forward(self, predict, target):
        pt = F.softmax(predict, dim=1)
        class_mask = F.one_hot(target, self.num_classes)
        ids = target.view(-1, 1)
        alpha = self.alpha[ids.data.view(-1)].to(predict.device)
        probs = (pt * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


