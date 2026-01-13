#Differentiable GPU ray projector

import torch
import torch.nn.functional as F

def rotate_volume(vol, theta):
    grid = F.affine_grid(
        torch.tensor([[[
            [torch.cos(theta), -torch.sin(theta), 0],
            [torch.sin(theta),  torch.cos(theta), 0]
        ]]], device=vol.device),
        vol.unsqueeze(0).unsqueeze(0).shape,
        align_corners=False
    )
    return F.grid_sample(vol.unsqueeze(0).unsqueeze(0), grid, align_corners=False)[0,0]

def forward_project(volume, angles):
    projections = []
    for a in angles:
        rot = rotate_volume(volume, torch.tensor(a, device=volume.device))
        proj = rot.sum(dim=0)
        projections.append(proj)
    return torch.stack(projections)
