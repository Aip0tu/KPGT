import os
import random
import numpy as np
import torch
import dgl
from dgl._ffi.base import DGLError

def _cuda_explicitly_disabled():
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    dgl_cuda_visible_devices = os.environ.get('DGL_CUDA_VISIBLE_DEVICES')
    disabled_values = {'', '-1'}
    return (
        cuda_visible_devices in disabled_values
        or dgl_cuda_visible_devices in disabled_values
    )

def set_random_seed(seed=22, n_threads=16, device_preference='auto'):
    """Set random seed.

    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    use_cuda = (
        device_preference != 'cpu'
        and torch.cuda.is_available()
        and not _cuda_explicitly_disabled()
    )
    if use_cuda:
        try:
            dgl.seed(seed)
            dgl.random.seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except DGLError:
            if device_preference == 'cuda':
                raise
            print("CUDA initialization failed, falling back to CPU.")
            use_cuda = False
    torch.set_num_threads(n_threads)
    os.environ['PYTHONHASHSEED'] = str(seed)
    return use_cuda
