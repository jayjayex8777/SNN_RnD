import torch
import torch.nn as nn

class LIFNode(nn.Module):
    def __init__(self, tau=2.0, threshold=0.5, reset_value=0.0):
        super().__init__()
        self.tau = tau
        self.threshold = threshold
        self.reset_value = reset_value
        self.register_buffer('v', None)

    def forward(self, x):
        if self.v is None or self.v.size() != x.size():
            self.v = torch.zeros_like(x)

        self.v = self.v + (x - self.v) / self.tau

        # Soft surrogate spike (TorchScript-compatible, no hard threshold)
        scale = 10.0
        out = scale * torch.sigmoid(scale * (self.v - self.threshold)) * (1 - torch.sigmoid(scale * (self.v - self.threshold)))

        self.v = torch.where(self.v >= self.threshold, torch.full_like(self.v, self.reset_value), self.v)
        return out

    def reset(self):
        self.v = None
