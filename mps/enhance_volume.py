import numpy as np
import torch
from models.unet import UNet
from src.utils import normalize, save_volume, get_device

device = get_device()
print("Using device:", device)

model = UNet().to(device)
model.load_state_dict(torch.load("denoiser.pt", map_location=device))
model.eval()

volume = normalize(np.load("recon.npy"))
enhanced = np.zeros_like(volume)

with torch.no_grad():
    for i in range(volume.shape[0]):
        inp = torch.tensor(volume[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        out = model(inp)
        enhanced[i] = out.squeeze().cpu().numpy()

        if i % 10 == 0:
            print(f"Processed slice {i}")

enhanced = normalize(enhanced)
save_volume(enhanced, "results/enhanced.npy")