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
#   scipy 1.0.0+
#   matplotlib 3.1.1+
#   ... other modules, see setup.py
# A collection of pyTorch modules, functions, optimizers,
# data processing tools and utilities. Could be used as an
# extension of pyTorch.
################################################################
# Update reports:
# ---------------
# 0.1.6 @ 8/6/2021
#   1. Support "thread_type" for data.h5py.H*Parser.
#   2. Fix a bug when GPU is absent for data.sequence.
# 0.1.5 @ 3/14/2021
#   1. Add DecoderNet to our standard module protocol.
#   2. Fix some bugs of data.h5py and data.preprocs.
#   3. Make draw.setFigure enhanced by contextlib.
#   4. Add a title in `Readme.md`.
#   5. Fix typos and bugs in `data` and `modules`.
#   6. Add properties `nlayers`, `input_size` for networks in `modules`.
# 0.1.2 @ 2/27/2021
#   1. Fix more feature problems in `contribs.torchsummary`.
#   2. Fix bugs and finish `data.preprocs`.
#   3. Add more features in `data.webtools`.
# 0.1.0 @ 2/26/2021
#   1. Create this project.
#   2. Add packages: `contribs`, `data`, `modules`, `utils`.
#   3. Finish `modules.conv`, `modules.resnet`.
#   4. Finish `data.h5py`, `data.webtools`.
#   5. Finish `contribs.torchsummary`.
#   6. Drop the plan for support `contribs.tqdm`, add
#      `utils.ContexWrapper` as for instead.
#   7. Add testing function for `data.webtools.DataChecker`.
################################################################
'''

# Import sub-modules
from . import contribs
from . import data
from . import modules
from . import utils

__version__ = '0.1.6'

__all__ = [
    'contribs', 'data', 'modules', 'utils'
]

# Set this local module as the prefered one
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

# Delete private sub-modules and objects
del extend_path
