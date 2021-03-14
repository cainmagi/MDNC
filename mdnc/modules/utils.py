#!python
# -*- coding: UTF-8 -*-
'''
################################################################
# Modules - Shared utilities
# @ Modern Deep Network Toolkits for pyTorch
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.5+
#   pyTorch 1.0.0+
# Share tools for computing module parameters for different
# module sets.
# These codes are private and should not be used by users.
################################################################
'''

import functools
import torch.nn as nn

__all__ = ['get_normalizer', 'get_upscaler', 'get_activator', 'get_adaptive_pooling',
           'check_is_stride', 'get_convnd', 'cal_kernel_padding', 'cal_scaled_shapes']


def get_normalizer(normalizer, order=2):
    if order == 3:
        if normalizer == 'inst':
            return nn.InstanceNorm3d
        elif normalizer == 'pinst':
            return functools.partial(nn.InstanceNorm3d, affine=True)
        else:
            return nn.BatchNorm3d
    elif order == 2:
        if normalizer == 'inst':
            return nn.InstanceNorm2d
        elif normalizer == 'pinst':
            return functools.partial(nn.InstanceNorm2d, affine=True)
        else:
            return nn.BatchNorm2d
    elif order == 1:
        if normalizer == 'inst':
            return nn.InstanceNorm1d
        elif normalizer == 'pinst':
            return functools.partial(nn.InstanceNorm1d, affine=True)
        else:
            return nn.BatchNorm1d
    else:
        raise ValueError('modules.utils: The argument "order" could only be 1, 2, or 3.')


def get_upscaler(stride, output_size=None, order=2):
    mode = 'trilinear' if order == 3 else ('bilinear' if order == 2 else 'linear')
    if output_size is None:
        if not isinstance(stride, (list, tuple)):
            stride = (stride, ) * order
        layer_upsample = nn.Upsample(scale_factor=stride, mode=mode, align_corners=True)
    else:
        layer_upsample = nn.Upsample(size=output_size, mode=mode, align_corners=True)
    return layer_upsample


def get_activator(activator, in_planes=None):
    if activator == 'prelu':
        return functools.partial(nn.PReLU, num_parameters=in_planes)
    elif activator == 'relu':
        return functools.partial(nn.ReLU, inplace=False)
    else:
        return None


def get_adaptive_pooling(order=2, out_size=1):
    if not isinstance(out_size, (list, tuple)):
        out_size = (out_size, ) * order
    if order == 3:
        return nn.AdaptiveAvgPool3d(out_size)
    elif order == 2:
        return nn.AdaptiveAvgPool2d(out_size)
    elif order == 1:
        return nn.AdaptiveAvgPool1d(out_size)
    else:
        raise ValueError('modules.utils: The argument "order" could only be 1, 2, or 3.')


def check_is_stride(stride, output_size=None, scaler='down'):
    if output_size is not None and scaler == 'up':
        return True
    if isinstance(stride, (list, tuple)):
        for s in stride:
            if s > 1:
                return True
        return False
    else:
        return stride > 1


def get_convnd(order=2):
    if order == 3:
        return nn.Conv3d
    elif order == 2:
        return nn.Conv2d
    elif order == 1:
        return nn.Conv1d
    else:
        raise ValueError('modules.utils: The argument "order" could only be 1, 2, or 3.')


def cal_kernel_padding(kernel_size, ksize_plus=0):
    if isinstance(kernel_size, (list, tuple)):
        ksize = list()
        psize = list()
        stride = list()
        for k in kernel_size:
            ks = k + ksize_plus if k > 1 else 1
            ksize.append(ks)
            psize.append(ks // 2)
            stride.append(2 if k > 1 else 1)
        return tuple(ksize), tuple(psize), tuple(stride)
    else:
        ksize = kernel_size + ksize_plus if kernel_size > 1 else 1
        psize = ksize // 2
        stride = 2 if kernel_size > 1 else 1
        return ksize, psize, stride


def cal_scaled_shapes(top_shape, level, stride=2):
    if not isinstance(top_shape, (list, tuple)):
        raise TypeError('modules.utils: The argument "top_shape" requires to be a sequence.')
    if (not isinstance(level, int)) or level <= 0:
        raise TypeError('modules.utils: The argument "level" requires to be a positive integer.')
    if not isinstance(stride, (list, tuple)):
        if isinstance(stride, int):
            stride = (stride, ) * len(top_shape)
        else:
            raise TypeError('modules.utils: The argument "stride" requires to be a sequence or integer.')

    def cal_down_scale(cur_shape):
        next_shape = list()
        for sh, st in zip(cur_shape, stride):
            v = sh // st
            if v == 0:
                raise ValueError('modules.utils: The shape {0} is too small, it could not be retrieved from the level {1}.'.format(top_shape, level))
            next_shape.append(v + (1 if sh % st > 0 else 0))
        return tuple(next_shape)

    shapes = [top_shape, ]
    cur_shape = top_shape
    for _ in range(level):
        cur_shape = cal_down_scale(cur_shape)
        shapes.append(cur_shape)
    return tuple(shapes)
