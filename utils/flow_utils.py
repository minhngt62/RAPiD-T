import torch
import torchvision.transforms.functional as tvf
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

def resize_flow_tensor(flow, out_size):
    # Only supprots constant scaling in both dimensions
    n_in, c_in, h_in, w_in = flow.shape
    w, h = out_size
    flow_out = tvf.resize(flow, [h, w])
    flow_out[:, 0, ...] *= w / w_in
    flow_out[:, 1, ...] *= h / h_in
    return flow_out