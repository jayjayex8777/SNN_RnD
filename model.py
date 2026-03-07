import torch
import torch.nn as nn
from lif_module import LIFNode


class SimpleSNN1d(nn.Module):
    """Conv1d-based SNN with soft surrogate LIF."""
    def __init__(self, in_channels=12, num_classes=4, T=20):
        super().__init__()
        self.T = T

        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.lif1 = LIFNode(tau=1.0, threshold=0.2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.lif2 = LIFNode(tau=0.9, threshold=0.15)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (B, C, N, T)
        xt = x[:, :, :, 0]       # (B, C, N)
        xt = self.conv1(xt)
        xt = self.lif1(xt)

        xt = self.conv2(xt)
        xt = self.lif2(xt)
        acc = xt.clone()

        for t in range(1, self.T):
            xt = x[:, :, :, t]    # (B, C, N)
            xt = self.conv1(xt)
            xt = self.lif1(xt)

            xt = self.conv2(xt)
            xt = self.lif2(xt)
            acc += xt

        out = acc / self.T
        out = self.pool(out).squeeze(-1)  # (B, 64)
        out = self.fc2(out)
        return out

    def reset(self):
        self.lif1.reset()
        self.lif2.reset()


# Hard spike + STE for Conv1d
class _HardSpikeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mem, threshold, width):
        ctx.save_for_backward(mem - threshold)
        ctx.width = width
        return (mem >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        (delta,) = ctx.saved_tensors
        mask = (delta.abs() <= ctx.width).float()
        grad_input = grad_output * mask / (2.0 * ctx.width + 1e-6)
        return grad_input, None, None


class HardLIFNode(nn.Module):
    def __init__(self, tau=2.0, threshold=0.5, surrogate_width=0.5, reset_value=0.0):
        super().__init__()
        self.tau = tau
        self.threshold = threshold
        self.surrogate_width = surrogate_width
        self.reset_value = reset_value
        self.register_buffer("v", torch.zeros(0))

    def forward(self, x):
        if self.v.numel() == 0 or self.v.shape != x.shape:
            self.v = torch.zeros_like(x)

        self.v = self.v + (x - self.v) / self.tau
        if torch.jit.is_scripting():
            spike = (self.v >= self.threshold).float()
        else:
            spike = _HardSpikeSTE.apply(self.v, self.threshold, self.surrogate_width)
        self.v = torch.where(spike > 0, torch.full_like(self.v, self.reset_value), self.v)
        return spike

    def reset(self):
        self.v = torch.zeros(0, device=self.v.device if isinstance(self.v, torch.Tensor) else None)


class StdpSNN1d(nn.Module):
    """Hard spike(0/1) + STE Conv1d SNN."""
    def __init__(self, in_channels=12, num_classes=4, T=20,
                 thresh1=0.2, thresh2=0.15, surrogate_width=0.5, gain=1.0):
        super().__init__()
        self.T = T
        self.gain = gain

        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.lif1 = HardLIFNode(tau=1.0, threshold=thresh1, surrogate_width=surrogate_width)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.lif2 = HardLIFNode(tau=0.9, threshold=thresh2, surrogate_width=surrogate_width)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        acc = torch.zeros(0, device=x.device)

        for t in range(self.T):
            xt = x[:, :, :, t]
            xt = self.conv1(xt) * self.gain
            xt = self.lif1(xt)

            xt = self.conv2(xt) * self.gain
            xt = self.lif2(xt)

            if acc.numel() == 0:
                acc = xt
            else:
                acc = acc + xt

        out = acc / float(self.T)
        out = self.pool(out).squeeze(-1)
        out = self.fc2(out)
        return out

    def reset(self):
        self.lif1.reset()
        self.lif2.reset()
