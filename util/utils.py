"""Util functions"""
import random

import torch
import numpy as np

CLASS_LABELS = {
    'VOC': {
        'all': set(range(1, 21)),
        0: set(range(1, 21)) - set(range(1, 6)),
        1: set(range(1, 21)) - set(range(6, 11)),
        2: set(range(1, 21)) - set(range(11, 16)),
        3: set(range(1, 21)) - set(range(16, 21)),
    },
    'COCO': {
        'all': set(range(1, 81)),
        0: set(range(1, 81)) - set(range(1, 21)),
        1: set(range(1, 81)) - set(range(21, 41)),
        2: set(range(1, 81)) - set(range(41, 61)),
        3: set(range(1, 81)) - set(range(61, 81)),
    }
}

def set_seed(seed):
    """
    Set the random seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)