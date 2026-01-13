import numpy as np

class ConeBeamGeometry:
    def __init__(self, DSO=500, DSD=800, det_pixels=256, pixel_size=1.0, n_angles=180):
        self.DSO = DSO
        self.DSD = DSD
        self.det_pixels = det_pixels
        self.pixel_size = pixel_size
        self.angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
