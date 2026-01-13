import numpy as np
import torch
from models.unet import UNet
from src.utils import normalize, save_volume, get_device

MODEL_PATH = "denoiser.pt"
INPUT_VOLUME = "recon.npy"
OUTPUT_VOLUME = "results/enhanced.npy"

def enhance():
    device = get_device()
    print(f"Using device: {device}")

    # Load model
    model = UNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Load volume
    volume = np.load(INPUT_VOLUME)
    volume = normalize(volume)

    enhanced = np.zeros_like(volume)

    # Slice-wise inference (axial)
    with torch.no_grad():
        for i in range(volume.shape[0]):
            slice_img = volume[i]
            inp = torch.tensor(slice_img).unsqueeze(0).unsqueeze(0).float().to(device)

            out = model(inp)
            enhanced[i] = out.squeeze().cpu().numpy()

            if i % 20 == 0:
                print(f"Processed slice {i}/{volume.shape[0]}")

    # Normalize and save
    enhanced = normalize(enhanced)
    save_volume(enhanced, OUTPUT_VOLUME)

    print(f"Enhanced volume saved to {OUTPUT_VOLUME}")

if __name__ == "__main__":
    enhance()
