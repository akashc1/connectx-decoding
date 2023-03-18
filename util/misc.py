import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_model(model, dst):
    torch.save(model.state_dict(), dst)


def grad_norm(model):
    return torch.cat([p.view(-1) for p in model.parameters()]).norm()
