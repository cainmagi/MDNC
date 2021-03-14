#!python
# -*- coding: UTF-8 -*-
'''
################################################################
# Modules
# @ Modern Deep Network Toolkits for pyTorch
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.5+
#   pyTorch 1.0.0+
# A collection of specially designed pyTorch modules, including
# special network layers and network models.
################################################################
# Update reports:
# ---------------
# 0.1.5 @ 3/2/2021
#   1. Add DecoderNet to conv, resnet.
#   2. Rename ConvNet by EncoderNet in conv, resnet.
#   3. Fix typos and bugs.
#   4. Add `nlayers` for all networks. Add `input_size` for
#      decoders.
# 0.1.0 @ 2/26/2021
#   1. Create sub-packages: conv, resnet.
################################################################
'''

# Import sub-modules
from . import conv
from . import resnet

__all__ = [
    'conv', 'resnet'
]

# Set this local module as the prefered one
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

# Delete private sub-modules and objects
del extend_path
