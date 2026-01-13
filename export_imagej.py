##File → Import → Raw → 128×128×128 uint16


import numpy as np

vol = np.load("recon.npy")
vol = (vol-vol.min())/(vol.max()-vol.min())
vol = (vol*65535).astype(np.uint16)
vol.tofile("recon.raw")
