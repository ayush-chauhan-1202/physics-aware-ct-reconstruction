import numpy as np, torch
from src.fdk import fdk
from src.geometry import ConeBeamGeometry

geom = ConeBeamGeometry()
projs = torch.tensor(np.load("data_projections.npy")).float().cuda()

recon = fdk(projs, geom.angles)
np.save("recon.npy", recon.cpu().numpy())
