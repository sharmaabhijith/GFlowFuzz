#!/usr/bin/env python3
# """
# Reproduction script for: Bug on Bernoulli (using GPU)
# API under test: torch.nn.functional
# """

import torch
import sys, platform, traceback
from torch.autograd import Variable
import torch.nn.functional as F

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
    a = F.sigmoid(torch.randn(10, 1))

    # This is OK
    a.bernoulli()

    # This is OK also
    b = a.cuda()
    b.bernoulli()

    # This is OK
    va = Variable(a)
    va.bernoulli()

    # This is a bug!
    vb = Variable(a).cuda()
    vb.bernoulli()

except Exception as e:
    print('\n[BUG TRIGGERED] Exception captured:')
    traceback.print_exc()
    sys.exit(1)

print('\n[FINISHED] Script completed without uncaught errors.')
