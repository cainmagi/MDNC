#!python
# -*- coding: UTF-8 -*-
'''
################################################################
# Data
# @ Modern Deep Network Toolkits for pyTorch
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.5+
#   urllib3 1.26.2+
#   numpy 1.13+
#   h5py 3.1.0+
# Optional package:
#   pyTorch 1.0.0+
# A collection of dataset loaders and data processing tools.
# The main modules includes:
#   h5py, ...
# They share similar APIs, and could be used for processing
# datasets by parallel. The architecture of them are inspired
# by keras.utils.Sequence. They could be used as substitutes
# of torch.utils.Dataset and torch.utils.DataLoader.
# The preprocs module could be used as preproc function of any
# parallel dataset manager. The preprocessing would be applied
# by parallel automatically.
# The webtools module is used for downloading online tarballs.
# Optional packages:
#   *
################################################################
# Update reports:
# ---------------
# 0.1.0 @ 2/25/2021
#   1. Create sub-packages: h5py, preprocs, sequence, webtools.
#   2. Finish h5py, webtools.
################################################################
'''

# Import sub-modules
from . import h5py
from . import preprocs
from . import sequence
from . import webtools

# Dynamically import the optional packages

__all__ = [
    'h5py', 'preprocs', 'sequence', 'webtools'
]

# Set this local module as the prefered one
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

# Delete private sub-modules and objects
del extend_path
