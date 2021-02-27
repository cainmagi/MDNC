#!python
# -*- coding: UTF-8 -*-
'''
################################################################
# Modern Deep Network Toolkits for pyTorch
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.5+
#   pyTorch 1.0.0+
#   numpy 1.13+
#   matplotlib 3.1.1+
#   ... other modules, see setup.py
# A collection of pyTorch modules, functions, optimizers,
# data processing tools and utilities. Could be used as an
# extension of pyTorch.
################################################################
# Update reports:
# ---------------
# 0.1.0 @ 2/26/2021
#   1. Create this project.
#   2. Add packages: `contribs`, `data`, `modules`, `utils`.
#   3. Finish `modules.conv`, `modules.resnet`.
#   4. Finish `data.h5py`, `data.webtools`.
#   5. Finish `contribs.torchsummary`.
################################################################
'''

# Import sub-modules
from . import contribs
from . import data
from . import modules
from . import utils

__version__ = '0.1.0'

__all__ = [
    'contribs', 'data', 'modules', 'utils'
]

# Set this local module as the prefered one
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

# Delete private sub-modules and objects
del extend_path
