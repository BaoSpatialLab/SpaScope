# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 20:01:54 2026

@author: bingbing & baobao
"""

import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Set the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    This function fixes random behavior across the SpaScope workflow, including
    graph construction, GAT inference, clustering initialization, and raster tie-breaking.
    It is recommended to call this function before running any SpaScope analysis.

    Parameters
    ----------
    seed : int, default=42
        Random seed used for Python's ``random`` module, NumPy, and PyTorch.

    Notes
    -----
    This function enables deterministic behavior in PyTorch as much as possible:
    - sets seeds for CPU and CUDA
    - disables cuDNN benchmark
    - enables deterministic algorithms

    Examples
    --------
    >>> from spascope import set_seed
    >>> set_seed(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)











