import torch
import torch.nn.functional as F

def rotate_slice(slice2d, theta):
    device = slice2d.device

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    affine = torch.tensor([
        [cos_t, -sin_t, 0],
        [sin_t,  cos_t, 0]
    ], device=device, dtype=torch.float32).unsqueeze(0)

    grid = F.affine_grid(
        affine,
        size=(1, 1, slice2d.shape[0], slice2d.shape[1]),
        align_corners=False
    )

    out = F.grid_sample(
        slice2d.unsqueeze(0).unsqueeze(0),
        grid,
        align_corners=False
    )

    return out[0, 0]

def forward_project(volume, angles):
    projections = []

    for a in angles:
        rotated_slices = []
        theta = torch.tensor(a, device=volume.device, dtype=torch.float32)

        for z in range(volume.shape[0]):
            rotated = rotate_slice(volume[z], theta)
            rotated_slices.append(rotated)

        rotated_volume = torch.stack(rotated_slices)
        proj = rotated_volume.sum(dim=0)
        projections.append(proj)

    return torch.stack(projections)