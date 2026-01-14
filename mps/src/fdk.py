import torch

def ramp_filter(p):
    freq = torch.fft.fftfreq(p.shape[-1], device=p.device)
    return torch.real(torch.fft.ifft(torch.fft.fft(p) * torch.abs(freq)))

def fdk(projections, angles):
    filtered = torch.stack([ramp_filter(p) for p in projections])

    D = projections.shape[-1]
    volume = torch.zeros((D, D, D), device=projections.device, dtype=torch.float32)

    for i, a in enumerate(angles):
        back = filtered[i].unsqueeze(0).repeat(D, 1, 1)
        k = int(-a * 180 / 3.1416 / 90) % 4
        volume += torch.rot90(back, k=k, dims=(1, 2))

    return volume / len(angles)