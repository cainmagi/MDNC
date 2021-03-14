#!python
# -*- coding: UTF-8 -*-
'''
################################################################
# Modules - residual network
# @ Modern Deep Network Toolkits for pyTorch
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.5+
#   pyTorch 1.0.0+
# This module is the definition of the residual network. The
# network could be initialized here and used for training and
# processing.
# The codes are inspired by:
#   https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
################################################################
'''

import functools

import torch
import torch.nn as nn

from .utils import get_convnd, get_normalizer, get_upscaler, get_activator, get_adaptive_pooling, check_is_stride, cal_kernel_padding, cal_scaled_shapes

__all__ = ['BlockPlain1d', 'BlockPlain2d', 'BlockPlain3d', 'BlockBottleneck1d', 'BlockBottleneck2d', 'BlockBottleneck3d',
           'UNet1d', 'UNet2d', 'UNet3d', 'unet16', 'unet32', 'unet44', 'unet65', 'unet83',
           'AE1d', 'AE2d', 'AE3d', 'ae16', 'ae32', 'ae44', 'ae65', 'ae83',
           'EncoderNet1d', 'EncoderNet2d', 'EncoderNet3d', 'encnet12', 'encnet32', 'encnet47', 'encnet62',
           'DecoderNet1d', 'DecoderNet2d', 'DecoderNet3d', 'decnet13', 'decnet33', 'decnet48', 'decnet63']


def get_block_depth(block):
    block = block.casefold()
    if block == 'bottleneck':
        return 3
    elif block == 'plain':
        return 2
    else:
        raise ValueError('module.resnet: The argument "block" should be "plain" or "bottleneck".')


class _BlockFactory:
    '''N-D block factory class for the residual network
    The block factory used for building any specialized block.
    '''
    def __init__(self, order, normalizer='pinst', activator='prelu', layer_order='new'):
        '''Initialization
        Arguments:
            order: the block dimension. For example, when order=2, the
                   nn.Conv2d would be used.
        Arguments (optional):
            normalizer: the normalization method, could be:
                        - "batch": Batch normalization.
                        - "inst": Instance normalization.
                        - "pinst": Instance normalization with tunable
                                    rescaling parameters.
                        - "null": Without normalization, would falls back to
                                    the "convolution + activation" form.
            activator: activation method, could be:
                       - "prelu", - "relu", - "null".
            layer_order: the sub-layer composition order, could be:
                         - "new": norm + activ + conv
                         - "old": conv + norm + activ
        '''
        super().__init__()
        if isinstance(order, int) and order in (1, 2, 3):
            self.order = order
        else:
            raise ValueError('modules.resnet: The argument "order" should be 1 or 2 or 3.')
        if isinstance(normalizer, str) and normalizer in ('batch', 'inst', 'pinst', 'null'):
            self.normalizer = normalizer
        else:
            raise ValueError('modules.resnet: The argument "normalizer" should be "batch" or "inst" or "pinst" or "null".')
        if isinstance(activator, str) and activator in ('relu', 'prelu', 'null'):
            self.activator = activator
        else:
            raise ValueError('modules.resnet: The argument "activator" should be "relu" or "prelu" or "null".')
        if isinstance(layer_order, str) and layer_order in ('old', 'new'):
            self.layer_order = layer_order
        else:
            raise ValueError('modules.resnet: The argument "layer_order" should be "old" or "new".')

    def res_branch(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, output_size=None, scaler='down'):
        '''Advanced Convolution branch.
        Arguments:
            in_planes: the channel number of the input data.
            out_planes: the channel number of the output data.
        Arguments (optional):
            kernel_size: the kernel size of this layer.
            stride: the stride size of this layer.
            padding: the padding size of the convolutional layer.
            output_size: the size of the output data. This option is only used
                         when "scaler=up". When setting this value, the size
                         of the up-sampling would be given explicitly and
                         the option "stride" would not be used.
            scaler: scaling method. Could be "down" or "up". When using "down",
                    the argument "stride" would be used for down-sampling; when
                    using "up", "stride" would be used for up-sampling
                    (equivalent to transposed convolution).
        Arguments (inherited):
            order, normalizer, activator, layer_order.
        '''
        ConvNd = get_convnd(order=self.order)
        is_stride = check_is_stride(stride, output_size=output_size, scaler=scaler)
        seq = []
        if self.normalizer == 'null':
            if (not is_stride) or scaler == 'down':
                seq.append(
                    ConvNd(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
                )
            elif is_stride and scaler == 'up':
                seq.extend((
                    get_upscaler(stride=stride, output_size=output_size, order=self.order),
                    ConvNd(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=padding, bias=True)
                ))
            else:
                raise ValueError('modules.resnet: The arguments "stride" and "scaler" should be valid.')
            new_activator = get_activator(self.activator, in_planes=out_planes)
            if new_activator is not None:
                seq.append(
                    new_activator()
                )
        elif self.normalizer in ('batch', 'inst', 'pinst'):
            normalizer_op = get_normalizer(self.normalizer, order=self.order)
            if self.layer_order == 'new':
                seq.append(
                    normalizer_op(in_planes)
                )
                new_activator = get_activator(self.activator, in_planes=in_planes)
                if new_activator is not None:
                    seq.append(
                        new_activator()
                    )
            if (not is_stride) or scaler == 'down':
                seq.append(
                    ConvNd(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
                )
            elif is_stride and scaler == 'up':
                seq.extend((
                    get_upscaler(stride=stride, output_size=output_size, order=self.order),
                    ConvNd(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
                ))
            else:
                raise ValueError('modules.resnet: The arguments "stride" and "scaler" should be valid.')
            if self.layer_order == 'old':
                seq.append(
                    normalizer_op(out_planes)
                )
                new_activator = get_activator(self.activator, in_planes=out_planes)
                if new_activator is not None:
                    seq.append(
                        new_activator()
                    )
        else:
            raise ValueError('modules.resnet: The arguments "normalizer"  should be valid.')
        return nn.Sequential(*seq)

    def in_branch(self, in_planes, out_planes, stride=1, output_size=None, scaler='down'):
        '''Input branch.
        Arguments:
            in_planes: the channel number of the input data.
            out_planes: the channel number of the output data.
        Arguments (optional):
            stride: the stride size of this layer.
            output_size: the size of the output data. This option is only used
                         when "scaler=up". When setting this value, the size
                         of the up-sampling would be given explicitly and
                         the option "stride" would not be used.
            scaler: scaling method. Could be "down" or "up". When using "down",
                    the argument "stride" would be used for down-sampling; when
                    using "up", "stride" would be used for up-sampling
                    (equivalent to transposed convolution).
        Arguments (inherited):
            order, normalizer.
        '''
        if check_is_stride(stride, output_size=output_size, scaler=scaler) or in_planes != out_planes:
            ConvNd = get_convnd(order=self.order)
            seq = []
            if self.normalizer in 'null':
                if scaler == 'down':
                    seq.append(ConvNd(in_planes, out_planes,
                               kernel_size=1, stride=stride, padding=0, bias=True))
                elif scaler == 'up':
                    seq.extend((
                        get_upscaler(stride=stride, output_size=output_size, order=self.order),
                        ConvNd(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True)
                    ))
                else:
                    raise ValueError('modules.resnet: The arguments "stride" and "scaler" should be valid.')
            elif self.normalizer in ('batch', 'inst', 'pinst'):
                normalizer_op = get_normalizer(self.normalizer, order=self.order)
                if scaler == 'down':
                    seq.append(ConvNd(in_planes, out_planes,
                               kernel_size=1, stride=stride, padding=0, bias=False))
                elif scaler == 'up':
                    seq.extend((
                        get_upscaler(stride=stride, output_size=output_size, order=self.order),
                        ConvNd(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
                    ))
                else:
                    raise ValueError('modules.resnet: The arguments "stride" and "scaler" should be valid.')
                seq.append(
                    normalizer_op(out_planes)
                )
            else:
                raise ValueError('modules.resnet: The arguments "normalizer"  should be valid.')
            return nn.Sequential(*seq)
        else:
            return None


class _BlockPlainNd(nn.Module):
    '''N-D plain residual block
    This is the implementation of the plain residual block. The residual
    block could be divided into two branches (input + conv). The convolutional
    branch is a composed of convN and convN.
    In the following paper, a new op composition order is proposed
    for building residual block:
        https://arxiv.org/abs/1603.05027
    We also support this implementation, set "layer_order=new" to enable it.
    '''

    def __init__(self, order, in_planes, out_planes, kernel_size=3, stride=1, padding=1, output_size=None,
                 normalizer='pinst', activator='prelu', layer_order='new', scaler='down'):
        '''Initialization
        Arguments:
            order: the block dimension. For example, when order=2, the
                   nn.Conv2d would be used.
            in_planes: the channel number of the input data.
            out_planes: the channel number of the output data.
        Arguments (optional):
            kernel_size: the kernel size of this layer.
            stride: the stride size of this layer.
            padding: the padding size of the residual block.
            output_size: the size of the output data. This option is only used
                         when "scaler=up". When setting this value, the size
                         of the up-sampling would be given explicitly and
                         the option "stride" would not be used.
            normalizer: the normalization method, could be:
                        - "batch": Batch normalization.
                        - "inst": Instance normalization.
                        - "pinst": Instance normalization with tunable
                                    rescaling parameters.
                        - "null": Without normalization, would falls back to
                                    the "convolution + activation" form.
            activator: activation method, could be:
                       - "prelu", - "relu", - "null".
            layer_order: the sub-layer composition order, could be:
                         - "new": norm + activ + conv
                         - "old": conv + norm + activ
            scaler: scaling method. Could be "down" or "up". When using "down",
                    the argument "stride" would be used for down-sampling; when
                    using "up", "stride" would be used for up-sampling
                    (equivalent to transposed convolution).
        '''
        super().__init__()
        res_factory = _BlockFactory(order, normalizer=normalizer, activator=activator, layer_order=layer_order)
        self.conv1 = res_factory.res_branch(in_planes, in_planes, kernel_size=kernel_size, padding=padding, stride=1)
        self.conv2 = res_factory.res_branch(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                            padding=padding, output_size=output_size, scaler=scaler)
        self.in_node = res_factory.in_branch(in_planes, out_planes, stride=stride, output_size=output_size, scaler=scaler)

    def forward(self, x):
        res = self.conv1(x)
        res = self.conv2(res)

        if self.in_node is not None:
            bch = self.in_node(x)
        else:
            bch = x

        res += bch
        return res


class _BlockBottleneckNd(nn.Module):
    '''N-D Bottleneck block
    This is the implementation of the bottleneck residual block. The residual
    block could be divided into two branches (input + conv). The convolutional
    branch is a composed of conv1, convN and conv1.
    In the following paper, a new op composition order is proposed
    for building residual block:
        https://arxiv.org/abs/1603.05027
    We also support this implementation, set "layer_order=new" to enable it.
    '''

    def __init__(self, order, in_planes, out_planes, kernel_size=3, stride=1, padding=1, output_size=None,
                 normalizer='pinst', activator='prelu', layer_order='new', scaler='down'):
        '''Initialization
        Arguments:
            order: the block dimension. For example, when order=2, the
                   nn.Conv2d would be used.
            in_planes: the channel number of the input data.
            out_planes: the channel number of the output data.
        Arguments (optional):
            kernel_size: the kernel size of this layer.
            stride: the stride size of this layer.
            padding: the padding size of the residual block.
            output_size: the size of the output data. This option is only used
                         when "scaler=up". When setting this value, the size
                         of the up-sampling would be given explicitly and
                         the option "stride" would not be used.
            normalizer: the normalization method, could be:
                        - "batch": Batch normalization.
                        - "inst": Instance normalization.
                        - "pinst": Instance normalization with tunable
                                    rescaling parameters.
                        - "null": Without normalization, would falls back to
                                    the "convolution + activation" form.
            activator: activation method, could be:
                       - "prelu", - "relu", - "null".
            layer_order: the sub-layer composition order, could be:
                         - "new": norm + activ + conv
                         - "old": conv + norm + activ
            scaler: scaling method. Could be "down" or "up". When using "down",
                    the argument "stride" would be used for down-sampling; when
                    using "up", "stride" would be used for up-sampling
                    (equivalent to transposed convolution).
        '''
        super().__init__()
        res_factory = _BlockFactory(order, normalizer=normalizer, activator=activator, layer_order=layer_order)
        self.conv1 = res_factory.res_branch(in_planes, in_planes, kernel_size=1, padding=0, stride=1)
        self.conv2 = res_factory.res_branch(in_planes, in_planes, kernel_size=kernel_size, stride=stride,
                                            padding=padding, output_size=output_size, scaler=scaler)
        self.conv3 = res_factory.res_branch(in_planes, out_planes, kernel_size=1, padding=0, stride=1)
        self.in_node = res_factory.in_branch(in_planes, out_planes, stride=stride, output_size=output_size, scaler=scaler)

    def forward(self, x):
        res = self.conv1(x)
        res = self.conv2(res)
        res = self.conv3(res)

        if self.in_node is not None:
            bch = self.in_node(x)
        else:
            bch = x

        res += bch
        return res


class _BlockResStkNd(nn.Module):
    '''Create the N-D stacked residual block.
    Each block contains several stacked residual blocks.
    This module is used for building UNetNd, AENd and Enc/DecoderNetNd,
    should not be used by users.
    '''
    def __init__(self, order, in_planes, out_planes, block='bottleneck',
                 hidden_planes=None, kernel_size=3, padding=1, stride=1,
                 stack_level=3, ex_planes=0, scaler='down', export_hidden=False):
        '''Initialization
        Arguments:
            order: the block dimension. For example, when order=2, the
                   nn.Conv2d would be used.
            in_planes: the channel number of the input data.
            out_planes: the channel number of the output data.
        Arguments (optional):
            block: the block type, could be:
                   - bottleneck, - plain
            hidden_planes: the channel number of the first hidden layer, would
                           also used as the base of the following channels. If
                           not set, would use "out_planes" as the default
                           value.
            kernel_size: the kernel size of the residual blocks.
            padding: the padding size of the residual blocks.
            stride: the stride size of the residual blocks.
            stack_level: the number of residual blocks in this block, requiring
                         to be >= 1.
            ex_planes: the channel number of the second input data. This value
                       is =0 in most time, but >0 during the decoding phase
                       of the U-Net. the extra input would be concatenated
                       with the input data.
            scaler: scaling method. Could be "down" or "up". When using "down",
                    the argument "stride" would be used for down-sampling; when
                    using "up", "stride" would be used for up-sampling
                    (equivalent to transposed convolution).
            export_hidden: whether to export the hidden layer as the second
                           output. This option is only used during the encoding
                           phase of the U-Net.
        '''
        super().__init__()
        if stack_level < 1:
            raise ValueError('modules.resnet: The argument "stack_level" should be at least one.')
        block = block.casefold()
        if block == 'bottleneck':
            Block = _BlockBottleneckNd
        elif block == 'plain':
            Block = _BlockPlainNd
        else:
            raise ValueError('module.resnet: The argument "block" should be "plain" or "bottleneck".')

        # The first layer performs the scaling.
        self.with_exinput = isinstance(ex_planes, int) and ex_planes > 0
        self.export_hidden = export_hidden
        hidden_planes = hidden_planes if (isinstance(hidden_planes, int) and hidden_planes > 0) else out_planes
        self.conv_list = nn.ModuleList()
        for i in range(stack_level - 1):
            self.conv_list.append(
                Block(order, (in_planes + ex_planes) if i == 0 else hidden_planes, hidden_planes,
                      kernel_size=kernel_size, padding=padding, stride=1, scaler='down')
            )
        self.conv_scale = Block(order, hidden_planes if stack_level > 1 else (in_planes + ex_planes), out_planes,
                                kernel_size=kernel_size, padding=padding, stride=stride, scaler=scaler)

    @staticmethod
    def cropping(x0, x1):
        x0_size = x0.shape[2:]
        x1_size = x1.shape[2:]
        x0_slice = [Ellipsis]
        x1_slice = [Ellipsis]
        for i in range(len(x0_size)):
            get_size = min(x0_size[i], x1_size[i])
            x0_shift = (x0_size[i] - get_size) // 2
            x1_shift = (x1_size[i] - get_size) // 2
            x0_slice.append(slice(x0_shift, x0_shift + get_size))
            x1_slice.append(slice(x1_shift, x1_shift + get_size))
        return torch.cat((x0[tuple(x0_slice)],
                          x1[tuple(x1_slice)]), dim=1)  # Middle cropping

    def forward(self, *x):
        if self.with_exinput:
            x = self.cropping(x[0], x[1])
        else:
            x = x[0]
        for layer in self.conv_list:
            x = layer(x)
        res = self.conv_scale(x)
        if self.export_hidden:
            return res, x
        else:
            return res


class _UNetNd(nn.Module):
    '''N-D residual network based U-Net
    This moule is a built-in model for residual U-Net. The network is
    inspired by:
        https://github.com/nikhilroxtomar/Deep-Residual-Unet
    The network would down-sample and up-sample the input data according to
    the network depth. The depth is given by the length of the argument
    "layers".
    '''
    def __init__(self, order, channel, layers, block='bottleneck', kernel_size=3, in_planes=1, out_planes=1):
        '''Initialization
        Arguments:
            order: the network dimension. For example, when order=2, the
                   nn.Conv2d would be used.
            channel: the channel number of the first layer, would also used
                     as the base of the following channels.
            layers: a list of layer numbers. Each number represents the number
                    of residual blocks of a stage. The stage numer, i.e.
                    the depth of the network is the length of this list.
        Arguments (optional):
            block: the block type, could be:
                   - bottleneck, - plain
            kernel_size: the kernel size of each block.
            in_planes: the channel number of the input data.
            out_planes: the channel number of the output data.
        '''
        super().__init__()
        if len(layers) < 2:
            raise ValueError('modules.resnet: The argument "layers" should contain at least 2 values, but provide "{0}"'.format(layers))
        self.__layers = layers
        self.__block_depth = get_block_depth(block)
        ConvNd = get_convnd(order=order)

        ksize_e, psize_e, _ = cal_kernel_padding(kernel_size, ksize_plus=2)
        self.conv_first = ConvNd(in_planes, channel, kernel_size=ksize_e, stride=1, padding=psize_e, bias=False)
        self.conv_down_list = nn.ModuleList()
        self.conv_up_list = nn.ModuleList()
        ksize, psize, stride = cal_kernel_padding(kernel_size)

        # Down scaling route
        self.conv_down_list.append(
            _BlockResStkNd(order, channel, channel, block=block, kernel_size=ksize, padding=psize,
                           stride=stride, stack_level=layers[0], ex_planes=0, scaler='down', export_hidden=True))
        for n_l in layers[1:-1]:
            self.conv_down_list.append(
                _BlockResStkNd(order, channel, channel * 2, block=block, kernel_size=ksize, padding=psize,
                               stride=stride, stack_level=n_l, ex_planes=0, scaler='down', export_hidden=True))
            channel = channel * 2

        # Middle block
        self.conv_middle_up = _BlockResStkNd(order, channel, channel, hidden_planes=channel * 2, block=block,
                                             kernel_size=ksize, padding=psize, stride=stride, stack_level=layers[-1], ex_planes=0, scaler='up')

        # Up scaling route
        for n_l in layers[-2:0:-1]:
            self.conv_up_list.append(
                _BlockResStkNd(order, channel, channel // 2, hidden_planes=channel, block=block,
                               kernel_size=ksize, padding=psize, stride=stride, stack_level=n_l, ex_planes=channel, scaler='up'))
            channel = channel // 2
        self.conv_up_list.append(
            _BlockResStkNd(order, channel, channel, hidden_planes=channel, block=block,
                           kernel_size=ksize, padding=psize, stride=1, stack_level=layers[0], ex_planes=channel, scaler='down'))
        self.conv_final = ConvNd(channel, out_planes, kernel_size=ksize_e, stride=1, padding=psize_e, bias=True)

    @property
    def nlayers(self):
        '''Return number of convolutional layers along the depth.
        '''
        if len(self.__layers) == 0:
            return 0
        n_layers = functools.reduce(lambda x, y: x + 2 * self.__block_depth * y, self.__layers[:-1], self.__block_depth * self.__layers[-1]) + 2
        return n_layers

    def forward(self, x):
        x = self.conv_first(x)
        x_down_route = list()
        for layer in self.conv_down_list:
            x, x_sk = layer(x)
            x_down_route.append(x_sk)
        x = self.conv_middle_up(x)
        x_down_route.reverse()
        for layer, x_sk in zip(self.conv_up_list, x_down_route):
            x = layer(x, x_sk)
        x = self.conv_final(x)
        return x


class _AENd(nn.Module):
    '''N-D residual auto-encoder.
    This moule is a built-in model for residual auto-encoder.
    The network would down-sample and up-sample and the input data according
    to the network depth. The depth is given by the length of the argument
    "layers".
    '''
    def __init__(self, order, channel, layers, block='bottleneck', kernel_size=3, in_planes=1, out_planes=1):
        '''Initialization
        Arguments:
            order: the network dimension. For example, when order=2, the
                   nn.Conv2d would be used.
            channel: the channel number of the first layer, would also used
                     as the base of the following channels.
            layers: a list of layer numbers. Each number represents the number
                    of residual blocks of a stage. The stage numer, i.e.
                    the depth of the network is the length of this list.
        Arguments (optional):
            block: the block type, could be:
                   - bottleneck, - plain
            kernel_size: the kernel size of each block.
            in_planes: the channel number of the input data.
            out_planes: the channel number of the output data.
        '''
        super().__init__()
        if len(layers) < 2:
            raise ValueError('modules.resnet: The argument "layers" should contain at least 2 values, but provide "{0}"'.format(layers))
        self.__layers = layers
        self.__block_depth = get_block_depth(block)
        ConvNd = get_convnd(order=order)

        ksize_e, psize_e, _ = cal_kernel_padding(kernel_size, ksize_plus=2)
        self.conv_first = ConvNd(in_planes, channel, kernel_size=ksize_e, stride=1, padding=psize_e, bias=False)
        self.conv_down_list = nn.ModuleList()
        self.conv_up_list = nn.ModuleList()
        ksize, psize, stride = cal_kernel_padding(kernel_size)

        # Down scaling route
        self.conv_down_list.append(
            _BlockResStkNd(order, channel, channel, block=block, kernel_size=ksize, padding=psize,
                           stride=stride, stack_level=layers[0], ex_planes=0, scaler='down'))
        for n_l in layers[1:-1]:
            self.conv_down_list.append(
                _BlockResStkNd(order, channel, channel * 2, block=block, kernel_size=ksize, padding=psize,
                               stride=stride, stack_level=n_l, ex_planes=0, scaler='down'))
            channel = channel * 2

        # Middle block
        self.conv_middle_up = _BlockResStkNd(order, channel, channel, hidden_planes=channel * 2, block=block,
                                             kernel_size=ksize, padding=psize, stride=stride, stack_level=layers[-1], scaler='up')

        # Up scaling route
        for n_l in layers[-2:0:-1]:
            self.conv_up_list.append(
                _BlockResStkNd(order, channel, channel // 2, hidden_planes=channel, block=block,
                               kernel_size=ksize, padding=psize, stride=stride, stack_level=n_l, scaler='up'))
            channel = channel // 2
        self.conv_up_list.append(
            _BlockResStkNd(order, channel, channel, hidden_planes=channel, block=block,
                           kernel_size=ksize, padding=psize, stride=1, stack_level=layers[0], scaler='down'))
        self.conv_final = ConvNd(channel, out_planes, kernel_size=ksize_e, stride=1, padding=psize_e, bias=True)

    @property
    def nlayers(self):
        '''Return number of convolutional layers along the depth.
        '''
        if len(self.__layers) == 0:
            return 0
        n_layers = functools.reduce(lambda x, y: x + 2 * self.__block_depth * y, self.__layers[:-1], self.__block_depth * self.__layers[-1]) + 2
        return n_layers

    @staticmethod
    def cropping(x, x_ref_s):
        x_size = x.shape[2:]
        x_ref_size = x_ref_s[2:]
        x_slice = [Ellipsis]
        for i in range(len(x_size)):
            get_size = min(x_size[i], x_ref_size[i])
            x_shift = (x_size[i] - get_size) // 2
            x_slice.append(slice(x_shift, x_shift + get_size))
        return x[tuple(x_slice)]  # Middle cropping

    def forward(self, x):
        x = self.conv_first(x)
        x_down_route = list()
        for layer in self.conv_down_list:
            x_down_route.append(x.shape)
            x = layer(x)
        x = self.conv_middle_up(x)
        x_down_route.reverse()
        for layer, x_ref_s in zip(self.conv_up_list, x_down_route):
            x = self.cropping(x, x_ref_s)
            x = layer(x)
        x = self.conv_final(x)
        return x


class _EncoderNetNd(nn.Module):
    '''N-D residual down-scale (encoder) network.
    This moule is a built-in model for residual network. The network could be
    used for down-scaling or classification.
    The network would down-sample and the input data according to the network
    depth. The depth is given by the length of the argument "layers".
    '''
    def __init__(self, order, channel, layers, block='bottleneck', kernel_size=3, in_planes=1, out_length=2):
        '''Initialization
        Arguments:
            order: the network dimension. For example, when order=2, the
                   nn.Conv2d would be used.
            channel: the channel number of the first layer, would also used
                     as the base of the following channels.
            layers: a list of layer numbers. Each number represents the number
                    of residual blocks of a stage. The stage numer, i.e.
                    the depth of the network is the length of this list.
        Arguments (optional):
            block: the block type, could be:
                   - bottleneck, - plain
            kernel_size: the kernel size of each block.
            in_planes: the channel number of the input data.
            out_length: the length of the output vector, if not set, the
                        output would not be flattened.
        '''
        super().__init__()
        if len(layers) < 2:
            raise ValueError('modules.resnet: The argument "layers" should contain at least 2 values, but provide "{0}"'.format(layers))
        self.__layers = layers
        self.__block_depth = get_block_depth(block)
        ConvNd = get_convnd(order=order)

        ksize_e, psize_e, _ = cal_kernel_padding(kernel_size, ksize_plus=2)
        ksize, psize, stride = cal_kernel_padding(kernel_size)
        # Down scaling route
        netbody = nn.ModuleList([
            ConvNd(in_planes, channel, kernel_size=ksize_e, stride=1, padding=psize_e, bias=False),
            _BlockResStkNd(order, channel, channel, block=block, kernel_size=ksize, padding=psize,
                           stride=stride, stack_level=layers[0], scaler='down')])
        for layer in layers[1:]:
            netbody.append(
                _BlockResStkNd(order, channel, channel * 2, block=block, kernel_size=ksize, padding=psize,
                               stride=stride, stack_level=layer, scaler='down'))
            channel = channel * 2
        netbody.append(
            ConvNd(channel, channel, kernel_size=ksize, stride=1, padding=psize, bias=True))
        self.is_out_vector = (out_length is not None and out_length > 0)
        if self.is_out_vector:
            netbody.append(get_adaptive_pooling(order=order, out_size=1))
        self.netbody = netbody
        if self.is_out_vector:
            self.fc = nn.Linear(channel, out_length, bias=True)

    @property
    def nlayers(self):
        '''Return number of convolutional layers along the depth.
        '''
        if len(self.__layers) == 0:
            return 0
        n_layers = functools.reduce(lambda x, y: x + self.__block_depth * y, self.__layers, 0) + 2
        return n_layers

    def forward(self, x):
        for layer in self.netbody:
            x = layer(x)
        if self.is_out_vector:
            x = torch.flatten(x, 1)
            return self.fc(x)
        else:
            return x


class _DecoderNetNd(nn.Module):
    '''N-D residual up-scale (decoder) network.
    This moule is a built-in model for residual network. The network could
    be used for up-scaling or generating samples.
    The network would up-sample and the input data according to the network
    depth. The depth is given by the length of the argument "layers".
    Different from the encoder network, this module requires the output
    shape, and the input shape is inferred from the given output shape.
    '''
    def __init__(self, order, channel, layers, out_size, block='bottleneck', kernel_size=3, in_length=2, out_planes=1):
        '''Initialization
        Arguments:
            order: the network dimension. For example, when order=2, the
                   nn.Conv2d would be used.
            channel: the channel number of the first layer, would also used
                     as the base of the following channels.
            layers: a list of layer numbers. Each number represents the number
                    of residual blocks of a stage. The stage numer, i.e.
                    the depth of the network is the length of this list.
            out_size: the shape of the output data (without the sample and
                      channel dimension). This argument is required, because
                      the shape of the first layer is inferred from it.
        Arguments (optional):
            block: the block type, could be:
                   - bottleneck, - plain
            kernel_size: the kernel size of each block.
            in_length: the length of the input vector, if not set, the
                       input requires to be a specific shape tensor. Use
                       the self.in_shape property to get it.
            out_planes: the channel number of the output data.
        '''
        super().__init__()
        if len(layers) < 2:
            raise ValueError('modules.conv: The argument "layers" should contain at least 2 values, but provide "{0}"'.format(layers))
        self.__layers = layers
        self.__block_depth = get_block_depth(block)
        self.__in_length = in_length
        ConvNd = get_convnd(order=order)
        ksize_e, psize_e, _ = cal_kernel_padding(kernel_size, ksize_plus=2)
        ksize, psize, stride = cal_kernel_padding(kernel_size)
        self.__order = order
        if isinstance(out_size, int):
            out_size = (out_size, ) * order
        self.shapes = cal_scaled_shapes(out_size, level=len(layers), stride=stride)
        channels = tuple(map(lambda n: channel * (2 ** n), range(len(layers) - 1, -1, -1)))
        self.__in_channel = channels[0]

        # Require to convert the vector into channels
        self.is_in_vector = (in_length is not None and in_length > 0)
        if self.is_in_vector:
            in_shapes = self.shapes[-1]
            in_pad = tuple(s - 1 for s in in_shapes)
            self.conv_in = ConvNd(in_length, channels[0], kernel_size=in_shapes, padding=in_pad, bias=True)

        # Up scaling route
        self.conv_first = ConvNd(channels[0], channels[0], kernel_size=ksize, stride=1, padding=psize, bias=False)
        netbody = nn.ModuleList()
        for i, layer in enumerate(layers[:-1]):
            netbody.append(
                _BlockResStkNd(order, channels[i], channels[i + 1], block=block, kernel_size=ksize, padding=psize,
                               stride=stride, stack_level=layer, scaler='up'))
        netbody.append(
            _BlockResStkNd(order, channels[-1], channels[-1], block=block, kernel_size=ksize, padding=psize,
                           stride=stride, stack_level=layers[-1], scaler='up'))
        self.netbody = netbody
        self.conv_last = ConvNd(channels[-1], out_planes, kernel_size=ksize_e, stride=1, padding=psize_e, bias=True)

    @property
    def input_size(self):
        if self.is_in_vector:
            return (self.__in_length, )
        else:
            return (self.__in_channel, *self.shapes[-1])

    @property
    def nlayers(self):
        '''Return number of convolutional layers along the depth.
        '''
        if len(self.__layers) == 0:
            return 0
        n_layers = functools.reduce(lambda x, y: x + self.__block_depth * y, self.__layers, 0) + (3 if self.is_in_vector else 2)
        return n_layers

    @staticmethod
    def cropping(x, x_ref_s):
        x_size = x.shape[2:]
        x_ref_size = x_ref_s
        x_slice = [Ellipsis]
        for i in range(len(x_size)):
            get_size = min(x_size[i], x_ref_size[i])
            x_shift = (x_size[i] - get_size) // 2
            x_slice.append(slice(x_shift, x_shift + get_size))
        return x[tuple(x_slice)]  # Middle cropping

    def forward(self, x):
        if self.is_in_vector:
            x = x[(Ellipsis, *((None,) * self.__order))]
            x = self.conv_in(x)
            x = self.cropping(x, self.shapes[-1])
        x = self.conv_first(x)
        for layer, xs in zip(self.netbody, self.shapes[-2::-1]):
            x = layer(x)
            x = self.cropping(x, xs)
        x = self.conv_last(x)
        return x


class BlockPlain1d(_BlockPlainNd):
    '''1D plain residual block
    This is the implementation of the plain residual block. The residual
    block could be divided into two branches (input + conv). The convolutional
    branch is a composed of convN and convN.
    In the following paper, a new op composition order is proposed
    for building residual block:
        https://arxiv.org/abs/1603.05027
    We also support this implementation, set "layer_order=new" to enable it.
    '''

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, output_size=None,
                 normalizer='pinst', activator='prelu', layer_order='new', scaler='down'):
        '''Initialization
        Arguments:
            in_planes: the channel number of the input data.
            out_planes: the channel number of the output data.
        Arguments (optional):
            kernel_size: the kernel size of this layer.
            stride: the stride size of this layer.
            padding: the padding size of the residual block.
            output_size: the size of the output data. This option is only used
                         when "scaler=up". When setting this value, the size
                         of the up-sampling would be given explicitly and
                         the option "stride" would not be used.
            normalizer: the normalization method, could be:
                        - "batch": Batch normalization.
                        - "inst": Instance normalization.
                        - "pinst": Instance normalization with tunable
                                    rescaling parameters.
                        - "null": Without normalization, would falls back to
                                    the "convolution + activation" form.
            activator: activation method, could be:
                       - "prelu", - "relu", - "null".
            layer_order: the sub-layer composition order, could be:
                         - "new": norm + activ + conv
                         - "old": conv + norm + activ
            scaler: scaling method. Could be "down" or "up". When using "down",
                    the argument "stride" would be used for down-sampling; when
                    using "up", "stride" would be used for up-sampling
                    (equivalent to transposed convolution).
        '''
        super().__init__(1, in_planes=in_planes, out_planes=out_planes, kernel_size=kernel_size,
                         stride=stride, padding=padding, output_size=output_size,
                         normalizer=normalizer, activator=activator,
                         layer_order=layer_order, scaler=scaler)


class BlockPlain2d(_BlockPlainNd):
    '''2D plain residual block
    This is the implementation of the plain residual block. The residual
    block could be divided into two branches (input + conv). The convolutional
    branch is a composed of convN and convN.
    In the following paper, a new op composition order is proposed
    for building residual block:
        https://arxiv.org/abs/1603.05027
    We also support this implementation, set "layer_order=new" to enable it.
    '''

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, output_size=None,
                 normalizer='pinst', activator='prelu', layer_order='new', scaler='down'):
        '''Initialization
        Arguments:
            in_planes: the channel number of the input data.
            out_planes: the channel number of the output data.
        Arguments (optional):
            kernel_size: the kernel size of this layer.
            stride: the stride size of this layer.
            padding: the padding size of the residual block.
            output_size: the size of the output data. This option is only used
                         when "scaler=up". When setting this value, the size
                         of the up-sampling would be given explicitly and
                         the option "stride" would not be used.
            normalizer: the normalization method, could be:
                        - "batch": Batch normalization.
                        - "inst": Instance normalization.
                        - "pinst": Instance normalization with tunable
                                    rescaling parameters.
                        - "null": Without normalization, would falls back to
                                    the "convolution + activation" form.
            activator: activation method, could be:
                       - "prelu", - "relu", - "null".
            layer_order: the sub-layer composition order, could be:
                         - "new": norm + activ + conv
                         - "old": conv + norm + activ
            scaler: scaling method. Could be "down" or "up". When using "down",
                    the argument "stride" would be used for down-sampling; when
                    using "up", "stride" would be used for up-sampling
                    (equivalent to transposed convolution).
        '''
        super().__init__(2, in_planes=in_planes, out_planes=out_planes, kernel_size=kernel_size,
                         stride=stride, padding=padding, output_size=output_size,
                         normalizer=normalizer, activator=activator,
                         layer_order=layer_order, scaler=scaler)


class BlockPlain3d(_BlockPlainNd):
    '''3D plain residual block
    This is the implementation of the plain residual block. The residual
    block could be divided into two branches (input + conv). The convolutional
    branch is a composed of convN and convN.
    In the following paper, a new op composition order is proposed
    for building residual block:
        https://arxiv.org/abs/1603.05027
    We also support this implementation, set "layer_order=new" to enable it.
    '''

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, output_size=None,
                 normalizer='pinst', activator='prelu', layer_order='new', scaler='down'):
        '''Initialization
        Arguments:
            in_planes: the channel number of the input data.
            out_planes: the channel number of the output data.
        Arguments (optional):
            kernel_size: the kernel size of this layer.
            stride: the stride size of this layer.
            padding: the padding size of the residual block.
            output_size: the size of the output data. This option is only used
                         when "scaler=up". When setting this value, the size
                         of the up-sampling would be given explicitly and
                         the option "stride" would not be used.
            normalizer: the normalization method, could be:
                        - "batch": Batch normalization.
                        - "inst": Instance normalization.
                        - "pinst": Instance normalization with tunable
                                    rescaling parameters.
                        - "null": Without normalization, would falls back to
                                    the "convolution + activation" form.
            activator: activation method, could be:
                       - "prelu", - "relu", - "null".
            layer_order: the sub-layer composition order, could be:
                         - "new": norm + activ + conv
                         - "old": conv + norm + activ
            scaler: scaling method. Could be "down" or "up". When using "down",
                    the argument "stride" would be used for down-sampling; when
                    using "up", "stride" would be used for up-sampling
                    (equivalent to transposed convolution).
        '''
        super().__init__(3, in_planes=in_planes, out_planes=out_planes, kernel_size=kernel_size,
                         stride=stride, padding=padding, output_size=output_size,
                         normalizer=normalizer, activator=activator,
                         layer_order=layer_order, scaler=scaler)


class BlockBottleneck1d(_BlockBottleneckNd):
    '''1D Bottleneck block
    This is the implementation of the bottleneck residual block. The residual
    block could be divided into two branches (input + conv). The convolutional
    branch is a composed of conv1, convN and conv1.
    In the following paper, a new op composition order is proposed
    for building residual block:
        https://arxiv.org/abs/1603.05027
    We also support this implementation, set "layer_order=new" to enable it.
    '''

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, output_size=None,
                 normalizer='pinst', activator='prelu', layer_order='new', scaler='down'):
        '''Initialization
        Arguments:
            in_planes: the channel number of the input data.
            out_planes: the channel number of the output data.
        Arguments (optional):
            kernel_size: the kernel size of this layer.
            stride: the stride size of this layer.
            padding: the padding size of the residual block.
            output_size: the size of the output data. This option is only used
                         when "scaler=up". When setting this value, the size
                         of the up-sampling would be given explicitly and
                         the option "stride" would not be used.
            normalizer: the normalization method, could be:
                        - "batch": Batch normalization.
                        - "inst": Instance normalization.
                        - "pinst": Instance normalization with tunable
                                    rescaling parameters.
                        - "null": Without normalization, would falls back to
                                    the "convolution + activation" form.
            activator: activation method, could be:
                       - "prelu", - "relu", - "null".
            layer_order: the sub-layer composition order, could be:
                         - "new": norm + activ + conv
                         - "old": conv + norm + activ
            scaler: scaling method. Could be "down" or "up". When using "down",
                    the argument "stride" would be used for down-sampling; when
                    using "up", "stride" would be used for up-sampling
                    (equivalent to transposed convolution).
        '''
        super().__init__(1, in_planes=in_planes, out_planes=out_planes, kernel_size=kernel_size,
                         stride=stride, padding=padding, output_size=output_size,
                         normalizer=normalizer, activator=activator,
                         layer_order=layer_order, scaler=scaler)


class BlockBottleneck2d(_BlockBottleneckNd):
    '''2D Bottleneck block
    This is the implementation of the bottleneck residual block. The residual
    block could be divided into two branches (input + conv). The convolutional
    branch is a composed of conv1, convN and conv1.
    In the following paper, a new op composition order is proposed
    for building residual block:
        https://arxiv.org/abs/1603.05027
    We also support this implementation, set "layer_order=new" to enable it.
    '''

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, output_size=None,
                 normalizer='pinst', activator='prelu', layer_order='new', scaler='down'):
        '''Initialization
        Arguments:
            in_planes: the channel number of the input data.
            out_planes: the channel number of the output data.
        Arguments (optional):
            kernel_size: the kernel size of this layer.
            stride: the stride size of this layer.
            padding: the padding size of the convolutional layer.
            output_size: the size of the output data. This option is only used
                         when "scaler=up". When setting this value, the size
                         of the up-sampling would be given explicitly and
                         the option "stride" would not be used.
            normalizer: the normalization method, could be:
                        - "batch": Batch normalization.
                        - "inst": Instance normalization.
                        - "pinst": Instance normalization with tunable
                                    rescaling parameters.
                        - "null": Without normalization, would falls back to
                                    the "convolution + activation" form.
            activator: activation method, could be:
                       - "prelu", - "relu", - "null".
            layer_order: the sub-layer composition order, could be:
                         - "new": norm + activ + conv
                         - "old": conv + norm + activ
            scaler: scaling method. Could be "down" or "up". When using "down",
                    the argument "stride" would be used for down-sampling; when
                    using "up", "stride" would be used for up-sampling
                    (equivalent to transposed convolution).
        '''
        super().__init__(2, in_planes=in_planes, out_planes=out_planes, kernel_size=kernel_size,
                         stride=stride, padding=padding, output_size=output_size,
                         normalizer=normalizer, activator=activator,
                         layer_order=layer_order, scaler=scaler)


class BlockBottleneck3d(_BlockBottleneckNd):
    '''3D Bottleneck block
    This is the implementation of the bottleneck residual block. The residual
    block could be divided into two branches (input + conv). The convolutional
    branch is a composed of conv1, convN and conv1.
    In the following paper, a new op composition order is proposed
    for building residual block:
        https://arxiv.org/abs/1603.05027
    We also support this implementation, set "layer_order=new" to enable it.
    '''

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, output_size=None,
                 normalizer='pinst', activator='prelu', layer_order='new', scaler='down'):
        '''Initialization
        Arguments:
            in_planes: the channel number of the input data.
            out_planes: the channel number of the output data.
        Arguments (optional):
            kernel_size: the kernel size of this layer.
            stride: the stride size of this layer.
            padding: the padding size of the residual block.
            output_size: the size of the output data. This option is only used
                         when "scaler=up". When setting this value, the size
                         of the up-sampling would be given explicitly and
                         the option "stride" would not be used.
            normalizer: the normalization method, could be:
                        - "batch": Batch normalization.
                        - "inst": Instance normalization.
                        - "pinst": Instance normalization with tunable
                                    rescaling parameters.
                        - "null": Without normalization, would falls back to
                                    the "convolution + activation" form.
            activator: activation method, could be:
                       - "prelu", - "relu", - "null".
            layer_order: the sub-layer composition order, could be:
                         - "new": norm + activ + conv
                         - "old": conv + norm + activ
            scaler: scaling method. Could be "down" or "up". When using "down",
                    the argument "stride" would be used for down-sampling; when
                    using "up", "stride" would be used for up-sampling
                    (equivalent to transposed convolution).
        '''
        super().__init__(3, in_planes=in_planes, out_planes=out_planes, kernel_size=kernel_size,
                         stride=stride, padding=padding, output_size=output_size,
                         normalizer=normalizer, activator=activator,
                         layer_order=layer_order, scaler=scaler)


class UNet1d(_UNetNd):
    '''1D residual network based U-Net
    This moule is a built-in model for residual U-Net. The network is
    inspired by:
        https://github.com/nikhilroxtomar/Deep-Residual-Unet
    The network would down-sample and up-sample the input data according to
    the network depth. The depth is given by the length of the argument
    "layers".
    '''
    def __init__(self, channel, layers, block='bottleneck', kernel_size=3, in_planes=1, out_planes=1):
        '''Initialization
        Arguments:
            channel: the channel number of the first layer, would also used
                     as the base of the following channels.
            layers: a list of layer numbers. Each number represents the number
                    of residual blocks of a stage. The stage numer, i.e.
                    the depth of the network is the length of this list.
        Arguments (optional):
            block: the block type, could be:
                   - bottleneck, - plain
            kernel_size: the kernel size of each block.
            in_planes: the channel number of the input data.
            out_planes: the channel number of the output data.
        '''
        super().__init__(1, channel=channel, layers=layers, block=block, kernel_size=kernel_size,
                         in_planes=in_planes, out_planes=out_planes)


class UNet2d(_UNetNd):
    '''2D residual network based U-Net
    This moule is a built-in model for residual U-Net. The network is
    inspired by:
        https://github.com/nikhilroxtomar/Deep-Residual-Unet
    The network would down-sample and up-sample the input data according to
    the network depth. The depth is given by the length of the argument
    "layers".
    '''
    def __init__(self, channel, layers, block='bottleneck', kernel_size=3, in_planes=1, out_planes=1):
        '''Initialization
        Arguments:
            channel: the channel number of the first layer, would also used
                     as the base of the following channels.
            layers: a list of layer numbers. Each number represents the number
                    of residual blocks of a stage. The stage numer, i.e.
                    the depth of the network is the length of this list.
        Arguments (optional):
            block: the block type, could be:
                   - bottleneck, - plain
            kernel_size: the kernel size of each block.
            in_planes: the channel number of the input data.
            out_planes: the channel number of the output data.
        '''
        super().__init__(2, channel=channel, layers=layers, block=block, kernel_size=kernel_size,
                         in_planes=in_planes, out_planes=out_planes)


class UNet3d(_UNetNd):
    '''3D residual network based U-Net
    This moule is a built-in model for residual U-Net. The network is
    inspired by:
        https://github.com/nikhilroxtomar/Deep-Residual-Unet
    The network would down-sample and up-sample the input data according to
    the network depth. The depth is given by the length of the argument
    "layers".
    '''
    def __init__(self, channel, layers, block='bottleneck', kernel_size=3, in_planes=1, out_planes=1):
        '''Initialization
        Arguments:
            channel: the channel number of the first layer, would also used
                     as the base of the following channels.
            layers: a list of layer numbers. Each number represents the number
                    of residual blocks of a stage. The stage numer, i.e.
                    the depth of the network is the length of this list.
        Arguments (optional):
            block: the block type, could be:
                   - bottleneck, - plain
            kernel_size: the kernel size of each block.
            in_planes: the channel number of the input data.
            out_planes: the channel number of the output data.
        '''
        super().__init__(3, channel=channel, layers=layers, block=block, kernel_size=kernel_size,
                         in_planes=in_planes, out_planes=out_planes)


class AE1d(_AENd):
    '''1D residual auto-encoder.
    This moule is a built-in model for residual auto-encoder.
    The network would down-sample and up-sample and the input data according
    to the network depth. The depth is given by the length of the argument
    "layers".
    '''
    def __init__(self, channel, layers, block='bottleneck', kernel_size=3, in_planes=1, out_planes=1):
        '''Initialization
        Arguments:
            channel: the channel number of the first layer, would also used
                     as the base of the following channels.
            layers: a list of layer numbers. Each number represents the number
                    of residual blocks of a stage. The stage numer, i.e.
                    the depth of the network is the length of this list.
        Arguments (optional):
            block: the block type, could be:
                   - bottleneck, - plain
            kernel_size: the kernel size of each block.
            in_planes: the channel number of the input data.
            out_planes: the channel number of the output data.
        '''
        super().__init__(1, channel=channel, layers=layers, block=block, kernel_size=kernel_size,
                         in_planes=in_planes, out_planes=out_planes)


class AE2d(_AENd):
    '''2D residual auto-encoder.
    This moule is a built-in model for residual auto-encoder.
    The network would down-sample and up-sample and the input data according
    to the network depth. The depth is given by the length of the argument
    "layers".
    '''
    def __init__(self, channel, layers, block='bottleneck', kernel_size=3, in_planes=1, out_planes=1):
        '''Initialization
        Arguments:
            channel: the channel number of the first layer, would also used
                     as the base of the following channels.
            layers: a list of layer numbers. Each number represents the number
                    of residual blocks of a stage. The stage numer, i.e.
                    the depth of the network is the length of this list.
        Arguments (optional):
            block: the block type, could be:
                   - bottleneck, - plain
            kernel_size: the kernel size of each block.
            in_planes: the channel number of the input data.
            out_planes: the channel number of the output data.
        '''
        super().__init__(2, channel=channel, layers=layers, block=block, kernel_size=kernel_size,
                         in_planes=in_planes, out_planes=out_planes)


class AE3d(_AENd):
    '''3D residual auto-encoder.
    This moule is a built-in model for residual auto-encoder.
    The network would down-sample and up-sample and the input data according
    to the network depth. The depth is given by the length of the argument
    "layers".
    '''
    def __init__(self, channel, layers, block='bottleneck', kernel_size=3, in_planes=1, out_planes=1):
        '''Initialization
        Arguments:
            channel: the channel number of the first layer, would also used
                     as the base of the following channels.
            layers: a list of layer numbers. Each number represents the number
                    of residual blocks of a stage. The stage numer, i.e.
                    the depth of the network is the length of this list.
        Arguments (optional):
            block: the block type, could be:
                   - bottleneck, - plain
            kernel_size: the kernel size of each block.
            in_planes: the channel number of the input data.
            out_planes: the channel number of the output data.
        '''
        super().__init__(3, channel=channel, layers=layers, block=block, kernel_size=kernel_size,
                         in_planes=in_planes, out_planes=out_planes)


class EncoderNet1d(_EncoderNetNd):
    '''1D residual down-scale (encoder) network.
    This moule is a built-in model for residual network. The network could be
    used for down-scaling or classification.
    The network would down-sample and the input data according to the network
    depth. The depth is given by the length of the argument "layers".
    '''
    def __init__(self, channel, layers, block='bottleneck', kernel_size=3, in_planes=1, out_length=2):
        '''Initialization
        Arguments:
            channel: the channel number of the first layer, would also used
                     as the base of the following channels.
            layers: a list of layer numbers. Each number represents the number
                    of residual blocks of a stage. The stage numer, i.e.
                    the depth of the network is the length of this list.
        Arguments (optional):
            block: the block type, could be:
                   - bottleneck, - plain
            kernel_size: the kernel size of each block.
            in_planes: the channel number of the input data.
            out_length: the length of the output vector, if not set, the
                        output would not be flattened.
        '''
        super().__init__(1, channel=channel, layers=layers, block=block, kernel_size=kernel_size,
                         in_planes=in_planes, out_length=out_length)


class EncoderNet2d(_EncoderNetNd):
    '''2D residual down-scale (encoder) network.
    This moule is a built-in model for residual network. The network could be
    used for down-scaling or classification.
    The network would down-sample and the input data according to the network
    depth. The depth is given by the length of the argument "layers".
    '''
    def __init__(self, channel, layers, block='bottleneck', kernel_size=3, in_planes=1, out_length=2):
        '''Initialization
        Arguments:
            channel: the channel number of the first layer, would also used
                     as the base of the following channels.
            layers: a list of layer numbers. Each number represents the number
                    of residual blocks of a stage. The stage numer, i.e.
                    the depth of the network is the length of this list.
        Arguments (optional):
            block: the block type, could be:
                   - bottleneck, - plain
            kernel_size: the kernel size of each block.
            in_planes: the channel number of the input data.
            out_length: the length of the output vector, if not set, the
                        output would not be flattened.
        '''
        super().__init__(2, channel=channel, layers=layers, block=block, kernel_size=kernel_size,
                         in_planes=in_planes, out_length=out_length)


class EncoderNet3d(_EncoderNetNd):
    '''3D residual down-scale (encoder) network.
    This moule is a built-in model for residual network. The network could be
    used for down-scaling or classification.
    The network would down-sample and the input data according to the network
    depth. The depth is given by the length of the argument "layers".
    '''
    def __init__(self, channel, layers, block='bottleneck', kernel_size=3, in_planes=1, out_length=2):
        '''Initialization
        Arguments:
            channel: the channel number of the first layer, would also used
                     as the base of the following channels.
            layers: a list of layer numbers. Each number represents the number
                    of residual blocks of a stage. The stage numer, i.e.
                    the depth of the network is the length of this list.
        Arguments (optional):
            block: the block type, could be:
                   - bottleneck, - plain
            kernel_size: the kernel size of each block.
            in_planes: the channel number of the input data.
            out_length: the length of the output vector, if not set, the
                        output would not be flattened.
        '''
        super().__init__(3, channel=channel, layers=layers, block=block, kernel_size=kernel_size,
                         in_planes=in_planes, out_length=out_length)


class DecoderNet1d(_DecoderNetNd):
    '''1D residual up-scale (decoder) network.
    This moule is a built-in model for residual network. The network
    could be used for up-scaling or generating samples.
    The network would up-sample and the input data according to the network
    depth. The depth is given by the length of the argument "layers".
    Different from the encoder network, this module requires the output
    shape, and the input shape is inferred from the given output shape.
    '''
    def __init__(self, channel, layers, out_size, block='bottleneck', kernel_size=3, in_length=2, out_planes=1):
        '''Initialization
        Arguments:
            channel: the channel number of the first layer, would also used
                     as the base of the following channels.
            layers: a list of layer numbers. Each number represents the number
                    of residual blocks of a stage. The stage numer, i.e.
                    the depth of the network is the length of this list.
            out_size: the shape of the output data (without the sample and
                      channel dimension). This argument is required, because
                      the shape of the first layer is inferred from it.
        Arguments (optional):
            kernel_size: the kernel size of each block.
            in_length: the length of the input vector, if not set, the
                       input requires to be a specific shape tensor. Use
                       the self.in_shape property to get it.
            out_planes: the channel number of the output data.
        '''
        super().__init__(1, channel=channel, layers=layers, out_size=out_size, block=block,
                         kernel_size=kernel_size, in_length=in_length, out_planes=out_planes)


class DecoderNet2d(_DecoderNetNd):
    '''2D residual up-scale (decoder) network.
    This moule is a built-in model for residual network. The network
    could be used for up-scaling or generating samples.
    The network would up-sample and the input data according to the network
    depth. The depth is given by the length of the argument "layers".
    Different from the encoder network, this module requires the output
    shape, and the input shape is inferred from the given output shape.
    '''
    def __init__(self, channel, layers, out_size, block='bottleneck', kernel_size=3, in_length=2, out_planes=1):
        '''Initialization
        Arguments:
            channel: the channel number of the first layer, would also used
                     as the base of the following channels.
            layers: a list of layer numbers. Each number represents the number
                    of residual blocks of a stage. The stage numer, i.e.
                    the depth of the network is the length of this list.
            out_size: the shape of the output data (without the sample and
                      channel dimension). This argument is required, because
                      the shape of the first layer is inferred from it.
        Arguments (optional):
            kernel_size: the kernel size of each block.
            in_length: the length of the input vector, if not set, the
                       input requires to be a specific shape tensor. Use
                       the self.in_shape property to get it.
            out_planes: the channel number of the output data.
        '''
        super().__init__(2, channel=channel, layers=layers, out_size=out_size, block=block,
                         kernel_size=kernel_size, in_length=in_length, out_planes=out_planes)


class DecoderNet3d(_DecoderNetNd):
    '''3D residual up-scale (decoder) network.
    This moule is a built-in model for residual network. The network
    could be used for up-scaling or generating samples.
    The network would up-sample and the input data according to the network
    depth. The depth is given by the length of the argument "layers".
    Different from the encoder network, this module requires the output
    shape, and the input shape is inferred from the given output shape.
    '''
    def __init__(self, channel, layers, out_size, block='bottleneck', kernel_size=3, in_length=2, out_planes=1):
        '''Initialization
        Arguments:
            channel: the channel number of the first layer, would also used
                     as the base of the following channels.
            layers: a list of layer numbers. Each number represents the number
                    of residual blocks of a stage. The stage numer, i.e.
                    the depth of the network is the length of this list.
            out_size: the shape of the output data (without the sample and
                      channel dimension). This argument is required, because
                      the shape of the first layer is inferred from it.
        Arguments (optional):
            kernel_size: the kernel size of each block.
            in_length: the length of the input vector, if not set, the
                       input requires to be a specific shape tensor. Use
                       the self.in_shape property to get it.
            out_planes: the channel number of the output data.
        '''
        super().__init__(3, channel=channel, layers=layers, out_size=out_size, block=block,
                         kernel_size=kernel_size, in_length=in_length, out_planes=out_planes)


def __get_unet_nd(order):
    if order == 3:
        return UNet3d
    elif order == 2:
        return UNet2d
    elif order == 1:
        return UNet1d
    else:
        raise ValueError('modules.resnet: The argument "order" could only be 1, 2, or 3.')


def __get_ae_nd(order):
    if order == 3:
        return AE3d
    elif order == 2:
        return AE2d
    elif order == 1:
        return AE1d
    else:
        raise ValueError('modules.resnet: The argument "order" could only be 1, 2, or 3.')


def __get_encnet_nd(order):
    if order == 3:
        return EncoderNet3d
    elif order == 2:
        return EncoderNet2d
    elif order == 1:
        return EncoderNet1d
    else:
        raise ValueError('modules.resnet: The argument "order" could only be 1, 2, or 3.')


def __get_decnet_nd(order):
    if order == 3:
        return DecoderNet3d
    elif order == 2:
        return DecoderNet2d
    elif order == 1:
        return DecoderNet1d
    else:
        raise ValueError('modules.resnet: The argument "order" could only be 1, 2, or 3.')


# unet
def unet16(order=2, **kwargs):
    '''Constructs a resnet.UNet-16 model.
    This model is equivalent to
        https://github.com/nikhilroxtomar/Deep-Residual-Unet
    Configurations:
        Network depth: 3
        Network block: plain
        Stage details: [2, 1, 1]
        First channel number: 64
    Arguments:
        order: the dimension of the network. For example, when
               order=2, the nn.Conv2d would be used.
    Other Arguments (see mdnc.modules.resnet.UNet*d):
        in_planes, out_planes, kernel_size
    '''
    model = __get_unet_nd(order)(64, [2, 1, 1], block='plain', **kwargs)
    return model


def unet32(order=2, **kwargs):
    '''Constructs a resnet.UNet-32 model.
    Configurations:
        Network depth: 3
        Network block: bottleneck
        Stage details: [2, 2, 2]
        First channel number: 64
    Arguments:
        order: the dimension of the network. For example, when
               order=2, the nn.Conv2d would be used.
    Other Arguments (see mdnc.modules.resnet.UNet*d):
        in_planes, out_planes, kernel_size
    '''
    model = __get_unet_nd(order)(64, [2, 2, 2], block='bottleneck', **kwargs)
    return model


def unet44(order=2, **kwargs):
    '''Constructs a resnet.UNet-44 model.
    Configurations:
        Network depth: 4
        Network block: bottleneck
        Stage details: [2, 2, 2, 2]
        First channel number: 64
    Arguments:
        order: the dimension of the network. For example, when
               order=2, the nn.Conv2d would be used.
    Other Arguments (see mdnc.modules.resnet.UNet*d):
        in_planes, out_planes, kernel_size
    '''
    model = __get_unet_nd(order)(64, [2, 2, 2, 2], block='bottleneck', **kwargs)
    return model


def unet65(order=2, **kwargs):
    '''Constructs a resnet.UNet-65 model.
    Configurations:
        Network depth: 4
        Network block: bottleneck
        Stage details: [3, 3, 3, 3]
        First channel number: 64
    Arguments:
        order: the dimension of the network. For example, when
               order=2, the nn.Conv2d would be used.
    Other Arguments (see mdnc.modules.resnet.UNet*d):
        in_planes, out_planes, kernel_size
    '''
    model = __get_unet_nd(order)(64, [3, 3, 3, 3], block='bottleneck', **kwargs)
    return model


def unet83(order=2, **kwargs):
    '''Constructs a resnet.UNet-83 model.
    Configurations:
        Network depth: 5
        Network block: bottleneck
        Stage details: [3, 3, 3, 3, 3]
        First channel number: 64
    Arguments:
        order: the dimension of the network. For example, when
               order=2, the nn.Conv2d would be used.
    Other Arguments (see mdnc.modules.resnet.UNet*d):
        in_planes, out_planes, kernel_size
    '''
    model = __get_unet_nd(order)(64, [3, 3, 3, 3, 3], block='bottleneck', **kwargs)
    return model


# ae
def ae16(order=2, **kwargs):
    '''Constructs a resnet.AE-16 model.
    Configurations:
        Network depth: 3
        Network block: plain
        Stage details: [2, 1, 1]
        First channel number: 64
    Arguments:
        order: the dimension of the network. For example, when
               order=2, the nn.Conv2d would be used.
    Other Arguments (see mdnc.modules.resnet.AE*d):
        in_planes, out_planes, kernel_size
    '''
    model = __get_ae_nd(order)(64, [2, 1, 1], block='plain', **kwargs)
    return model


def ae32(order=2, **kwargs):
    '''Constructs a resnet.AE-32 model.
    Configurations:
        Network depth: 3
        Network block: bottleneck
        Stage details: [2, 2, 2]
        First channel number: 64
    Arguments:
        order: the dimension of the network. For example, when
               order=2, the nn.Conv2d would be used.
    Other Arguments (see mdnc.modules.resnet.AE*d):
        in_planes, out_planes, kernel_size
    '''
    model = __get_ae_nd(order)(64, [2, 2, 2], block='bottleneck', **kwargs)
    return model


def ae44(order=2, **kwargs):
    '''Constructs a resnet.AE-44 model.
    Configurations:
        Network depth: 4
        Network block: bottleneck
        Stage details: [2, 2, 2, 2]
        First channel number: 64
    Arguments:
        order: the dimension of the network. For example, when
               order=2, the nn.Conv2d would be used.
    Other Arguments (see mdnc.modules.resnet.AE*d):
        in_planes, out_planes, kernel_size
    '''
    model = __get_ae_nd(order)(64, [2, 2, 2, 2], block='bottleneck', **kwargs)
    return model


def ae65(order=2, **kwargs):
    '''Constructs a resnet.AE-65 model.
    Configurations:
        Network depth: 4
        Network block: bottleneck
        Stage details: [3, 3, 3, 3]
        First channel number: 64
    Arguments:
        order: the dimension of the network. For example, when
               order=2, the nn.Conv2d would be used.
    Other Arguments (see mdnc.modules.resnet.AE*d):
        in_planes, out_planes, kernel_size
    '''
    model = __get_ae_nd(order)(64, [3, 3, 3, 3], block='bottleneck', **kwargs)
    return model


def ae83(order=2, **kwargs):
    '''Constructs a resnet.AE-83 model.
    Configurations:
        Network depth: 5
        Network block: bottleneck
        Stage details: [3, 3, 3, 3, 3]
        First channel number: 64
    Arguments:
        order: the dimension of the network. For example, when
               order=2, the nn.Conv2d would be used.
    Other Arguments (see mdnc.modules.resnet.AE*d):
        in_planes, out_planes, kernel_size
    '''
    model = __get_ae_nd(order)(64, [3, 3, 3, 3, 3], block='bottleneck', **kwargs)
    return model


# encnet
def encnet12(order=2, **kwargs):
    '''Constructs a resnet.EncoderNet-12 model.
    Configurations:
        Network depth: 5
        Network block: plain
        Stage details: [1, 1, 1, 1, 1]
        First channel number: 64
    Arguments:
        order: the dimension of the network. For example, when
               order=2, the nn.Conv2d would be used.
    Other Arguments (see mdnc.modules.resnet.EncoderNet*d):
        in_planes, out_length, kernel_size
    '''
    model = __get_encnet_nd(order)(64, [1, 1, 1, 1, 1], block='plain', **kwargs)
    return model


def encnet32(order=2, **kwargs):
    '''Constructs a resnet.EncoderNet-32 model.
    Configurations:
        Network depth: 5
        Network block: bottleneck
        Stage details: [2, 2, 2, 2, 2]
        First channel number: 64
    Arguments:
        order: the dimension of the network. For example, when
               order=2, the nn.Conv2d would be used.
    Other Arguments (see mdnc.modules.resnet.EncoderNet*d):
        in_planes, out_length, kernel_size
    '''
    model = __get_encnet_nd(order)(64, [2, 2, 2, 2, 2], block='bottleneck', **kwargs)
    return model


def encnet47(order=2, **kwargs):
    '''Constructs a resnet.EncoderNet-47 model.
    Configurations:
        Network depth: 5
        Network block: bottleneck
        Stage details: [3, 3, 3, 3, 3]
        First channel number: 64
    Arguments:
        order: the dimension of the network. For example, when
               order=2, the nn.Conv2d would be used.
    Other Arguments (see mdnc.modules.resnet.EncoderNet*d):
        in_planes, out_length, kernel_size
    '''
    model = __get_encnet_nd(order)(64, [3, 3, 3, 3, 3], block='bottleneck', **kwargs)
    return model


def encnet62(order=2, **kwargs):
    '''Constructs a resnet.EncoderNet-62 model.
    Configurations:
        Network depth: 5
        Network block: bottleneck
        Stage details: [4, 4, 4, 4, 4]
        First channel number: 64
    Arguments:
        order: the dimension of the network. For example, when
               order=2, the nn.Conv2d would be used.
    Other Arguments (see mdnc.modules.resnet.EncoderNet*d):
        in_planes, out_length, kernel_size
    '''
    model = __get_encnet_nd(order)(64, [4, 4, 4, 4, 4], block='bottleneck', **kwargs)
    return model


# decnet
def decnet13(out_size, order=2, **kwargs):
    '''Constructs a resnet.DecoderNet-13 model.
    Configurations:
        Network depth: 5
        Network block: plain
        Stage details: [1, 1, 1, 1, 1]
        First channel number: 64
    Arguments:
        out_size: the output shape of the network.
        order: the dimension of the network. For example, when
               order=2, the nn.Conv2d would be used.
    Other Arguments (see mdnc.modules.resnet.DecoderNet*d):
        in_length, out_planes, kernel_size
    '''
    model = __get_decnet_nd(order)(64, [1, 1, 1, 1, 1], out_size=out_size, block='plain', **kwargs)
    return model


def decnet33(out_size, order=2, **kwargs):
    '''Constructs a conv.DecoderNet-33 model.
    Configurations:
        Network depth: 5
        Network block: bottleneck
        Stage details: [2, 2, 2, 2, 2]
        First channel number: 64
    Arguments:
        out_size: the output shape of the network.
        order: the dimension of the network. For example, when
               order=2, the nn.Conv2d would be used.
    Other Arguments (see mdnc.modules.resnet.DecoderNet*d):
        in_length, out_planes, kernel_size
    '''
    model = __get_decnet_nd(order)(64, [2, 2, 2, 2, 2], out_size=out_size, block='bottleneck', **kwargs)
    return model


def decnet48(out_size, order=2, **kwargs):
    '''Constructs a conv.DecoderNet-48 model.
    Configurations:
        Network depth: 5
        Network block: bottleneck
        Stage details: [3, 3, 3, 3, 3]
        First channel number: 64
    Arguments:
        out_size: the output shape of the network.
        order: the dimension of the network. For example, when
               order=2, the nn.Conv2d would be used.
    Other Arguments (see mdnc.modules.resnet.DecoderNet*d):
        in_length, out_planes, kernel_size
    '''
    model = __get_decnet_nd(order)(64, [3, 3, 3, 3, 3], out_size=out_size, block='bottleneck', **kwargs)
    return model


def decnet63(out_size, order=2, **kwargs):
    '''Constructs a conv.DecoderNet-63 model.
    Configurations:
        Network depth: 5
        Network block: bottleneck
        Stage details: [4, 4, 4, 4, 4]
        First channel number: 64
    Arguments:
        out_size: the output shape of the network.
        order: the dimension of the network. For example, when
               order=2, the nn.Conv2d would be used.
    Other Arguments (see mdnc.modules.resnet.DecoderNet*d):
        in_length, out_planes, kernel_size
    '''
    model = __get_decnet_nd(order)(64, [4, 4, 4, 4, 4], out_size=out_size, block='bottleneck', **kwargs)
    return model
