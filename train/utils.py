import numpy as np
import os
import random
import torch

def seed_everything(seed, eps=10):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determenistic = True
    torch.backends.cudnn.benchmark = False
    torch.set_printoptions(precision=eps)
