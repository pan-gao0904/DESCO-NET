import torch

def tv_loss(x):
    """Total Variation loss — encourages spatial smoothness.

    Used to regularize PGAM's α and β maps: color correction
    parameters should vary smoothly in space.
    """
    dx = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]).mean()
    dy = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]).mean()
    return dx + dy

