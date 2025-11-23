"""
Implementation 6

https://arxiv.org/abs/2302.00670
"""

from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
torch.backends.cudnn.benchmark = True

from implementations.base_implementation import (
    UNet, sample_loop, cosine_beta_schedule)