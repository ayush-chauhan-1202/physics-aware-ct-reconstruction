# from src.phantom import generate_sphere_phantom
# from src.projector import forward_project
# from src.geometry import ConeBeamGeometry
# import torch, numpy as np

# geom = ConeBeamGeometry()
# phantom = generate_sphere_phantom(128)
# vol = torch.tensor(phantom).float().cuda()

# projs = forward_project(vol, geom.angles)
# np.save("data_projections.npy", projs.cpu().numpy())
# np.save("gt_volume.npy", phantom)



import numpy as np
from src.phantom import generate_sphere_phantom
from src.projector import forward_project
from src.geometry import ConeBeamGeometry
from src.utils import get_device, to_tensor

device = get_device()
print("Using device:", device)

geom = ConeBeamGeometry()
phantom = generate_sphere_phantom(size=96)

vol = to_tensor(phantom, device)
projs = forward_project(vol, geom.angles)

np.save("data_projections.npy", projs.cpu().numpy())
np.save("gt_volume.npy", phantom)