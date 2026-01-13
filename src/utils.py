
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from pathlib import Path

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize(volume):
    vmin, vmax = volume.min(), volume.max()
    return (volume - vmin) / (vmax - vmin + 1e-8)

def save_volume(volume, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, volume)

def load_volume(path):
    return np.load(path)

def show_slices(volume, title="Slices"):
    """Visualize central axial/coronal/sagittal slices"""
    z, y, x = np.array(volume.shape) // 2

    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs[0].imshow(volume[z], cmap="gray")
    axs[0].set_title("Axial")

    axs[1].imshow(volume[:, y, :], cmap="gray")
    axs[1].set_title("Coronal")

    axs[2].imshow(volume[:, :, x], cmap="gray")
    axs[2].set_title("Sagittal")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def to_tensor(volume, device=None):
    device = device or get_device()
    return torch.tensor(volume, dtype=torch.float32, device=device)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()
