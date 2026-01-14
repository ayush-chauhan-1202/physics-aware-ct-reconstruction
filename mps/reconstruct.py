import numpy as np
from src.fdk import fdk
from src.iterative import sirt
from src.geometry import ConeBeamGeometry
from src.utils import get_device, to_tensor

device = get_device()
print("Using device:", device)

# Load geometry and projections
geom = ConeBeamGeometry()
projs = np.load("data_projections.npy")
projs_t = to_tensor(projs, device)

# -------- FDK Reconstruction --------
print("Running FDK reconstruction...")
recon_fdk = fdk(projs_t, geom.angles)
np.save("recon_fdk.npy", recon_fdk.cpu().numpy())
print("Saved recon_fdk.npy")

# -------- Iterative Reconstruction (SIRT) --------
print("Running iterative reconstruction (SIRT)...")
recon_sirt = sirt(projs_t, geom.angles, vol_shape = (96, 96, 96), iters=30, step_size=1e-2)
np.save("recon_sirt.npy", recon_sirt.cpu().numpy())
print("Saved recon_sirt.npy")