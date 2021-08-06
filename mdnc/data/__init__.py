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
#   scipy 1.0.0+
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
# 0.1.6 @ 8/6/2021
#   1. Support "thread_type" for h5py.H*Parser.
#   2. Fix a bug when GPU is absent for sequence.
# 0.1.5 @ 3/14/2021
#   1. Fix typos and bugs.
# 0.1.2 @ 2/27/2021
#   1. Finish preprocs.
#      - Update the implementation of preprocs.ProcAbstract.
#      - Add preprocs.ProcMerge,
#        preprocs.ProcFilter1d, preprocs.ProcNSTFilter1d,
#        preprocs.Pad, preprocs.ProcLifter.
#      - Fix a fatal bug in preprocs.ProcAbstract. The stack
#        order in the previous version is not corrected.
#   2. Add more features to webtools.
#      - Add safe control codes for urllib3.PoolManager in the
#        webtools module.
#      - Arrange the following functions:
#        - webtools.download_tarball
#        - webtools.download_tarball_public
#        - webtools.download_tarball_private
#        - webtools.download_tarball_link
#        Now the download_tarball() could downloads data from
#        both public repositories and private repositories,
#        and only requires the token when needing it.
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
