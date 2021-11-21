import torch.nn as nn

def featEmbedding(in_, out_):
    '''
    in_: input channel, e.g. 32
    out_: output channel, e.g. 64
    k: kernel size, e.g. 3 or (3,3)
    s: stride, e.g. 1 or (1,1)
    '''
    return nn.Sequential(
        nn.Conv2d(in_, 512, (1, 1), (1, 1), padding=0, bias=True),
        nn.LeakyReLU(0.1),
        nn.Conv2d(512, 512, (3, 3), (1, 1), padding=1, bias=True),
        nn.LeakyReLU(0.1),
        nn.Conv2d(512, out_, (1, 1), (1, 1), padding=0, bias=True),
    )