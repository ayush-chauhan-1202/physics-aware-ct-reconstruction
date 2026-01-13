import numpy as np

def generate_sphere_phantom(size=128):
    vol = np.zeros((size,size,size))
    cx = cy = cz = size//2
    for x in range(size):
        for y in range(size):
            for z in range(size):
                if (x-cx)**2 + (y-cy)**2 + (z-cz)**2 < (size//4)**2:
                    vol[x,y,z] = 1.0
    return vol
