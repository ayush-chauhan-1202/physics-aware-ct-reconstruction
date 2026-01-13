import torch
from .projector import forward_project

def sirt(projections, angles, iters=50, lr=1e-2):
    D = projections.shape[-1]
    vol = torch.zeros((D,D,D), device=projections.device, requires_grad=True)
    opt = torch.optim.Adam([vol], lr=lr)

    for i in range(iters):
        opt.zero_grad()
        loss = ((forward_project(vol, angles)-projections)**2).mean()
        loss.backward()
        opt.step()
        if i%10==0:
            print(f"Iter {i} loss {loss.item():.5f}")
    return vol.detach()
