import numpy as np

def psnr(x,y):
    mse = ((x-y)**2).mean()
    return 20*np.log10(1.0/np.sqrt(mse))
