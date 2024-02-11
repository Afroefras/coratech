from torch import Tensor


class TorchStandardScaler:
    def fit(self, x: Tensor) -> Tensor:
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)

    def transform(self, x: Tensor) -> Tensor:
        x -= self.mean
        x /= self.std + 1e-7
        return x
