import torch, numpy as np
from models.unet import UNet

x = torch.tensor(np.load("recon.npy")).float().unsqueeze(1)
y = torch.tensor(np.load("gt_volume.npy")).float().unsqueeze(1)

model = UNet().cuda()
opt = torch.optim.Adam(model.parameters(),1e-3)

for e in range(50):
    pred = model(x.cuda())
    loss = ((pred-y.cuda())**2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    print(e, loss.item())

torch.save(model.state_dict(),"denoiser.pt")
