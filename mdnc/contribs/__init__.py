#!python
# -*- coding: UTF-8 -*-
'''
################################################################
# Contribs
# @ Modern Deep Network Toolkits for pyTorch
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.5+
#   pyTorch 1.0.0+
# A collection of third-party packages. All of the packages are
# modified for bug fixing or the compatibility of MDNC.
################################################################
# Update reports:
# ---------------
# 0.1.2 @ 2/27/2021
#   1. Enhance the torchsummary module by:
#      - Fix bugs caused by "batch_size" and "dtypes".
#      - Add text auto wrap.
#      - Fix the parameter counting in general cases.
#      - Fix the size overflow problem.
#      - Add docstring.
# 0.1.0 @ 2/25/2021
#   1. Create sub-packages: torchsummary.
################################################################
'''

# Import sub-modules
from . import torchsummary

__all__ = [
    'torchsummary'
]

# Set this local module as the prefered one
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

# Delete private sub-modules and objects
del extend_path
