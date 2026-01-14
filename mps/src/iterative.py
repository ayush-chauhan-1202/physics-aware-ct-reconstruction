import torch
from src.projector import forward_project


def backproject(projections, angles, vol_shape):
    """
    Simple adjoint of forward projector (approximate backprojection)
    projections: [num_angles, H, W]
    returns: volume [D, H, W]
    """
    device = projections.device
    D, H, W = vol_shape

    volume = torch.zeros((D, H, W), device=device)

    for i, angle in enumerate(angles):
        proj = projections[i]  # [H, W]

        # Expand projection back into volume
        expanded = proj.unsqueeze(0).repeat(D, 1, 1)

        # Rotate each slice back
        rotated_slices = []
        for z in range(D):
            rotated = rotate_slice(expanded[z], -angle, device)
            rotated_slices.append(rotated)

        volume += torch.stack(rotated_slices)

    return volume / len(angles)


def rotate_slice(slice2d, angle, device):
    """
    Safe 2D rotation using affine_grid + grid_sample (forward only, no grad)
    """
    import torch.nn.functional as F

    theta = torch.tensor(angle, device=device, dtype=torch.float32)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    affine = torch.tensor([
        [cos_t, -sin_t, 0],
        [sin_t,  cos_t, 0]
    ], device=device).unsqueeze(0)

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


def sirt(projections, angles, vol_shape=(96, 96, 96), iters=20, step_size=1e-2):
    """
    Classical SIRT iterative reconstruction.
    No autograd. Fully MPS-safe.
    """
    device = projections.device
    volume = torch.zeros(vol_shape, device=device)

    print(f"Starting SIRT on device: {device}")
    print(f"Iterations: {iters}")

    for it in range(iters):

        # Forward projection
        sim_proj = forward_project(volume, angles)

        # Residual
        error = projections - sim_proj
        loss = torch.mean(error ** 2).item()

        # Backprojection of residual
        correction = backproject(error, angles, vol_shape)

        # Update
        volume = volume + step_size * correction

        if it % 5 == 0 or it == iters - 1:
            print(f"Iter {it:03d} | MSE: {loss:.6f}")

    return volume