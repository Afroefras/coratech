from torch import Tensor

def standard_scale(x: Tensor) -> Tensor:
    x_mean = x.mean()
    x_std = x.std() + 1e-10

    scaled = x.clone()
    scaled -= x_mean
    scaled /= x_std
    
    return scaled

def min_max_scale(x: Tensor) -> Tensor:
    x_min = x.min()
    x_max = x.max()

    scaled = x.clone()
    scaled -= x_min
    scaled /= x_max - x_min

    return scaled