from torch import Tensor

def standard_scale(x: Tensor) -> Tensor:
    x_mean = x.mean()
    x_std = x.std() + 1e-10

    scaled = x.clone()
    scaled -= x_mean
    scaled /= x_std
    
    return scaled
