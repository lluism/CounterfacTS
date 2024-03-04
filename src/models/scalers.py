from abc import ABC
from typing import Optional

import torch


def calculate_masked_mean(x: torch.Tensor, dim: int = 1, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Calculate the dimension wise mean of the input tensor ignoring masked values
    """
    num_observed = torch.sum(mask, dim=dim)
    num_observed[num_observed == 0] = 1  # set batches with a mean of 0 or no observations to 1
    sum_observed = torch.sum(x * mask, dim=dim)
    return (sum_observed / num_observed).unsqueeze(dim=dim)


class Scaler(ABC):
    def __init__(self, device: torch.device):
        self.device = device

    def fit(self, x: torch.Tensor) -> None:
        raise NotImplementedError

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MeanAbsScaler(Scaler):
    def __init__(self, device: torch.device, eps: float = 1.0) -> None:
        super().__init__(device)
        self.scale = None
        self.eps = eps

    def fit(self, x: torch.Tensor, dim: int = 1, mask: Optional[torch.Tensor] = None) -> None:
        if mask is None:
            mask = torch.ones_like(x, device=x.device)

        self.scale = calculate_masked_mean(x, dim=dim, mask=mask).clamp(min=self.eps)

    def fit_transform(self, x: torch.Tensor, dim: int = 1, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.fit(x, dim=dim, mask=mask)
        return self.transform(x)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.scale

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.size()) == 4:
            scale = self.scale.unsqueeze(dim=1).repeat(1, x.size(1), 1, 1)
        else:
            scale = self.scale

        return x * scale


class StandardScaler(Scaler):
    def __init__(self, device: torch.device, eps: float = 1.0) -> None:
        super().__init__(device)
        self.mean = None
        self.std = None
        self.eps = eps

    def fit(self, x: torch.Tensor, dim: int = 1, mask: Optional[torch.Tensor] = None) -> None:
        if mask is None:
            mask = torch.ones_like(x, device=x.device)

        self.mean = (calculate_masked_mean(x, dim=dim, mask=mask))

        num_observed = torch.sum(mask, dim=dim)
        num_observed[num_observed == 0] = 1
        var = torch.sum(torch.square((x - self.mean) * mask), dim=dim) / num_observed

        self.std = torch.sqrt(var).unsqueeze(dim=dim).clamp(min=self.eps)

    def fit_transform(self, x: torch.Tensor, dim: int = 1, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.fit(x, dim=dim, mask=mask)
        return self.transform(x, mask=mask)

    def transform(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            return ((x - self.mean) / self.std) * mask

        return (x - self.mean) / self.std

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        # check if we get [batch_size, num_samples, context_length, 1] as input
        if len(x.size()) == 4:
            mean = self.mean.unsqueeze(dim=1).repeat(1, x.size(1), 1, 1)
            std = self.std.unsqueeze(dim=1).repeat(1, x.size(1), 1, 1)
        else:
            mean = self.mean
            std = self.std

        return x * std + mean


class IdentityScaler(Scaler):
    def __init__(self, device: torch.device) -> None:
        super().__init__(device)

    def fit(self, x: torch.Tensor, dim: int = 1) -> None:
        pass

    def fit_transform(self, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        return x

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x


def get_scaler(scaler: str) -> Scaler:
    scalers = {
        "standard": StandardScaler,
        "mean_abs": MeanAbsScaler,
        "none": IdentityScaler
    }
    return scalers[scaler]
