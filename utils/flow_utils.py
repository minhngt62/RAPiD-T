import torch
from torch.nn.functional import grid_sample

def warp_feat(x, flow):
    n, c, h, w = x.shape
    flow_map = -resize_flow_tensor(flow, (w, h))
    flow_map[:, 0 , ...] += torch.arange(w).cuda()
    flow_map[:, 1 , ...] += torch.arange(h).view(-1, 1).cuda()
    flow_map[:, 0 , ...] = (flow_map[:, 0 , ...] * 2 / w) - 1
    flow_map[:, 1 , ...] = (flow_map[:, 1 , ...] * 2 / h) - 1
    y = grid_sample(x, flow_map.permute(0, 2, 3, 1), padding_mode="border")
    return y