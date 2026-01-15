[gen_data.tif](https://github.com/user-attachments/files/24642865/gen_data.tif)# Physics-Aware Cone Beam CT Reconstruction

A GPU-accelerated, physics-informed CT reconstruction pipeline combining:
- Classical filtered backprojection (FDK)
- Iterative reconstruction (SIRT via autodiff)
- Deep learning enhancement (U-Net)
- Synthetic CT simulation pipeline
- ImageJ-compatible export

## Motivation
Industrial imaging systems (X-ray CT, NDE) require high-quality reconstruction under noise, sparse views, and physics constraints. This project explores hybrid reconstruction combining physics and learning.

## Features
- Cone-beam style forward projector
- FDK implementation from scratch
- Iterative reconstruction using PyTorch autodiff
- Synthetic phantom generation
- DL-based post reconstruction enhancement
- Export to ImageJ raw volumes

## Example Results
| Ground Truth | FDK | FDK + DL | Iterative |
|--------------|-----|-----------|-------|
|<img width="386" height="384" alt="image" src="https://github.com/user-attachments/assets/b5c20557-c6f7-4bfe-ba65-dd473637752a" />
              |     |           |       |

## How to Run
```bash
pip install -r requirements.txt
python generate_data.py
python reconstruct.py
python export_imagej.py
