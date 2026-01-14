import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from pathlib import Path

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def normalize(volume):
    vmin, vmax = volume.min(), volume.max()
    return (volume - vmin) / (vmax - vmin + 1e-8)

def save_volume(volume, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, volume)

def load_volume(path):
    return np.load(path)

def to_tensor(x, device):
    return torch.tensor(x, dtype=torch.float32, device=device)

def to_numpy(x):
    return x.detach().cpu().numpy()

def show_slices(volume, title="Slices"):
    z, y, x = np.array(volume.shape) // 2
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs[0].imshow(volume[z], cmap="gray"); axs[0].set_title("Axial")
    axs[1].imshow(volume[:, y, :], cmap="gray"); axs[1].set_title("Coronal")
    axs[2].imshow(volume[:, :, x], cmap="gray"); axs[2].set_title("Sagittal")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()