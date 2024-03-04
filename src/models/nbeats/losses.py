import torch

from .utils import divide_no_nan


def mase_loss(insample: torch.Tensor, freq: int,
              forecast: torch.Tensor, target: torch.Tensor) -> torch.float:
    """
    MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf
    :param insample: Insample values. Shape: batch, time_i
    :param freq: Frequency value
    :param forecast: Forecast values. Shape: batch, time_o
    :param target: Target values. Shape: batch, time_o
    :return: Loss value
    """
    mask = torch.ones_like(forecast)  # unlike the original implementation, we only sample full windows
    masep = torch.mean(torch.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
    masked_masep_inv = divide_no_nan(mask, masep[:, None])
    return torch.mean(torch.abs(target - forecast) * masked_masep_inv)


def mape_loss(forecast: torch.Tensor, target: torch.Tensor) -> torch.float:
    """
    MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    :param forecast: Forecast values. Shape: batch, time
    :param target: Target values. Shape: batch, time
    :param mask: 0/1 mask. Shape: batch, time
    :return: Loss value
    """
    mask = torch.ones_like(forecast)
    weights = divide_no_nan(mask, target)
    return torch.mean(torch.abs((forecast - target) * weights))


def smape_2_loss(forecast: torch.Tensor, target: torch.Tensor) -> torch.float:
    """
    sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)
    :param forecast: Forecast values. Shape: batch, time
    :param target: Target values. Shape: batch, time
    :param mask: 0/1 mask. Shape: batch, time
    :return: Loss value
    """
    mask = torch.ones_like(forecast)
    return 200 * torch.mean(divide_no_nan(torch.abs(forecast - target),
                                          torch.abs(forecast.data) + torch.abs(target.data)) * mask)
