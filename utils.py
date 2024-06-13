import torch
import torch.nn as nn


class SoftSort(torch.nn.Module):
    def __init__(self, bit, hard=True, pow=1.0):
        super(SoftSort, self).__init__()
        self.hard = hard
        self.tau = bit
        self.pow = pow

    def forward(self, s):
        s = s.unsqueeze(-1)
        s_sorted = s.sort(descending=True, dim=1)[0]
        pairwise_diff = (s.transpose(1, 2) - s_sorted).abs().pow(self.pow).neg() / self.tau
        p = pairwise_diff.softmax(-1)
        if self.hard:
            s = torch.zeros_like(p, device=p.device)
            s.scatter_(-1, p.topk(1, -1)[1], value = 1)
            p = (s - p).detach() + p
        return p


class SortedNCELoss(nn.Module):
    def __init__(self, positive_num, tau, batch_size):
        super(SortedNCELoss, self).__init__()
        self.positive_num = positive_num
        self.batch_size = batch_size
        self.tau = tau
        self.softmax = nn.Softmax(dim=1)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, cos, labels):
        loss = 0
        for i in range(self.positive_num):
            pos, neg = cos[:, i], cos[:, self.positive_num:]
            logit = self.softmax(torch.cat([pos.unsqueeze(1), neg], dim=1) / self.tau)
            loss += self.cross_entropy(logit, labels) / self.positive_num / self.batch_size
        return loss


def data_supply(batch, images, batch_size):
    while images.shape[0] < batch_size:
        num_to_pad = batch_size - images.shape[0]
        images_pad, labels, indices = batch
        num = min(num_to_pad, batch_size)
        images_pad = images_pad[:num]
        images = torch.cat([images, images_pad], dim=0)
    return images