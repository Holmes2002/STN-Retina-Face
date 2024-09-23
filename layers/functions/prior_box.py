import torch
from itertools import product as product
import numpy as np
from math import ceil


class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
if __name__ == "__main__":
    cfg = {
    'name': 'mobilev3',
    # 'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'anchor_num': 6,
    'min_sizes': [[54, 33], [111, 54], [150, 53], [102, 81], [151, 74], [147, 144], [219, 98], [249, 120], [322, 144]],
    'steps': [8, 8, 8, 16, 16, 16, 32, 32, 32],
    'variance': [0.1, 0.2],
    'clip': True,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 8,
    'ngpu': 1,
    'epoch': 350,
    'decay1': 190,
    'decay2': 220,
    'image_size': 680,
    'pretrain': False,
    'return_layers': {'bneck1': 1, 'bneck2': 2, 'bneck3': 3},
    'in_channel': 32,
    'out_channel': 64
}

    img_dim = cfg['image_size']
    anchors = PriorBox(cfg, (img_dim, img_dim)).forward()
    print(anchors.shape)