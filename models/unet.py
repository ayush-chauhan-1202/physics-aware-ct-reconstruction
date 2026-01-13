import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1,32,3,1,1), nn.ReLU(),
            nn.Conv2d(32,64,3,2,1), nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64,32,2,2), nn.ReLU(),
            nn.Conv2d(32,1,1)
        )

    def forward(self,x):
        return self.dec(self.enc(x))
