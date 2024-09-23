import cv2
import torch
import numpy as np
from tqdm import tqdm
from STN.dataset import *
from torchvision import transforms
import os
import argparse
from utils import utils
import torch
from torchvision import transforms as T
from STN.model import STNet, Advanced_STNet
import torch.nn.functional as F
import torch.nn as nn

model = Advanced_STNet()
# model.load_state_dict(torch.load('checkpoint/71_26.pt'))
model.to('cuda')

