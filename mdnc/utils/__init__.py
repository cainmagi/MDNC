#!python
# -*- coding: UTF-8 -*-
'''
################################################################
# Utilities
# @ Modern Deep Network Toolkits for pyTorch
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.5+
#   numpy 1.13+
#   matplotlib 3.1.1+
# A collection of data processing or visualization tools not
# related to datasets or pyTorch.
################################################################
# Update reports:
# ---------------
# 0.1.5 @ 3/9/2021
#   1. Enhance the draw.setFigure by using contextlib.
# 0.1.0 @ 2/25/2021
#   1. Create sub-packages: draw, tools.
################################################################
'''

# Import sub-modules
from . import draw
from . import tools

__all__ = [
    'draw', 'tools'
]

# Set this local module as the prefered one
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

# Delete private sub-modules and objects
del extend_path
