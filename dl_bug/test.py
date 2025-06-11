#!/usr/bin/env python3
"""
Reproduction script for: Segmentation Fault: torch.view_as_complex fails with segfault for a zero dimensional tensor
API under test: torch.view_as_complexBug Description: Segmentation Fault: torch.view_as_complex fails with segfault for a zero dimensional tensor
"""

import torch
import sys, platform, traceback

# -- Environment diagnostics --
print('Platform  :', platform.platform())
print('Python    :', sys.version.split()[0])
print('PyTorch   :', torch.__version__)
print('CUDA avail:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA ver  :', torch.version.cuda)
    print('Device    :', torch.cuda.get_device_name(0))

# -- Deterministic seeds for reproducibility --
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# ---------- Original snippet begins (executed inside try/except) ----------
try:
    import torch

    t = torch.tensor(1.0, dtype=torch.float32)
    print(t.is_complex())
    t.view_as_complex()
except Exception as e:
    print('\n[BUG TRIGGERED] Exception captured:')
    traceback.print_exc()
    sys.exit(1)

print('\n[FINISHED] Script completed without uncaught errors.')