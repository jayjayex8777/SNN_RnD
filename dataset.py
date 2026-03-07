import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.special import expit


def rate_code_zscore_sigmoid(df: pd.DataFrame, T: int = 20):
    assert all(col in df.columns for col in ["ax", "ay", "az", "gx", "gy", "gz"]), \
        "6-axis sensor column is needed."

    channels = []
    spikes = []

    for col in ["ax", "ay", "az", "gx", "gy", "gz"]:
        signal = df[col].to_numpy()

        pos = np.clip(signal, 0, None)
        neg = np.clip(-signal, 0, None)

        for s, suffix in zip([pos, neg], ["_pos", "_neg"]):
            mean = np.mean(s)
            std = np.std(s) + 1e-8
            z = (s - mean) / std
            prob = expit(z)
            spike_seq = (np.random.rand(len(s), T) < prob[:, None]).astype(np.uint8)
            spikes.append(spike_seq)
            channels.append(col + suffix)

    spike_tensor = np.stack(spikes, axis=1)
    return spike_tensor, channels


class FileLevelSpikeDataset(Dataset):
    def __init__(self, folder, spike_func, T=20):
        self.samples = []
        self.spike_func = spike_func
        self.T = T

        for file in os.listdir(folder):
            if not file.endswith(".csv"):
                continue
            path = os.path.join(folder, file)

            if "swipe_up" in file and "sensor" in file:
                label = 0
            elif "swipe_down" in file and "sensor" in file:
                label = 1
            elif "flick_up" in file and "sensor" in file:
                label = 2
            elif "flick_down" in file and "sensor" in file:
                label = 3
            else:
                continue

            self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        df = pd.read_csv(path)
        spike_tensor, _ = self.spike_func(df, T=self.T)  # (N, 12, T)
        spike_tensor = spike_tensor.transpose(1, 0, 2)   # (12, N, T)
        return torch.tensor(spike_tensor, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def pad_collate(batch):
    """Pad to max sequence length. Output: (B, 12, N_max, T)."""
    xs, ys = zip(*batch)
    max_n = max(x.shape[1] for x in xs)
    c, _, t = xs[0].shape
    out = torch.zeros(len(xs), c, max_n, t, dtype=xs[0].dtype)
    for i, x in enumerate(xs):
        n = x.shape[1]
        out[i, :, :n, :] = x
    return out, torch.tensor(ys, dtype=torch.long)
