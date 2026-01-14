import numpy as np

vol = np.load("results/enhanced.npy")
vol = (vol - vol.min()) / (vol.max() - vol.min())
vol = (vol * 65535).astype(np.uint16)

vol.tofile("results/enhanced.raw")
print("Saved results/enhanced.raw (Import in ImageJ as uint16, size 96x96x96)")