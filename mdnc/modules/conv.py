#!python
# -*- coding: UTF-8 -*-
'''
################################################################
# Modules - convolutional network
# @ Modern Deep Network Toolkits for pyTorch
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.5+
#   pyTorch 1.0.0+
# This module is the definition of the convolutional network.
# The network could be initialized here and used for training
# and processing.
################################################################
'''

import functools

import torch
import torch.nn as nn

from .utils import get_convnd, get_normalizer, get_upscaler, get_activator, get_adaptive_pooling, check_is_stride, cal_kernel_padding, cal_scaled_shapes

__all__ = ['ConvModern1d', 'ConvModern2d', 'ConvModern3d',
           'UNet1d', 'UNet2d', 'UNet3d', 'unet12', 'unet16', 'unet17', 'unet23', 'unet29',
           'AE1d', 'AE2d', 'AE3d', 'ae12', 'ae16', 'ae17', 'ae23', 'ae29',
           'EncoderNet1d', 'EncoderNet2d', 'EncoderNet3d', 'encnet12', 'encnet15', 'encnet17', 'encnet22',
           'DecoderNet1d', 'DecoderNet2d', 'DecoderNet3d', 'decnet13', 'decnet16', 'decnet18', 'decnet23']


class _ConvModernNd(nn.Module):
    '''N-D Modern convolutional layer
    The implementation for the modern convolutional layer. The modern
    convolutional layer is a stack of convolution, normalization and
    activation.
    In the following paper, a new op composition order is proposed
    for building residual block:
        https://arxiv.org/abs/1603.05027
    We also support this implementation.
    '''
    def __init__(self, order, in_planes, out_planes, kernel_size=3, stride=1, padding=1, output_size=None,
                 normalizer='pinst', activator='prelu', layer_order='new', scaler='down'):
        '''Initialization.
        Arguments:
            order: the block dimension. For example, when order=2, the
                   nn.Conv2d would be used.
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
        super().__init__()
        ConvNd = get_convnd(order=order)
        is_stride = check_is_stride(stride, output_size=output_size, scaler=scaler)
        seq = []
        if normalizer == 'null':
            if (not is_stride) or scaler == 'down':
                seq.append(
                    ConvNd(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
                )
            elif is_stride and scaler == 'up':
                seq.extend((
                    get_upscaler(stride=stride, output_size=output_size, order=order),
                    ConvNd(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=padding, bias=True)
                ))
            else:
                raise ValueError('modules.conv: The arguments "stride" and "scaler" should be valid.')
            new_activator = get_activator(activator, in_planes=out_planes)
            if new_activator is not None:
                seq.append(
                    new_activator()
                )
        elif normalizer in ('batch', 'inst', 'pinst'):
            normalizer_op = get_normalizer(normalizer, order=order)
            if layer_order == 'new':
                seq.append(
                    normalizer_op(in_planes)
                )
                new_activator = get_activator(activator, in_planes=in_planes)
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
                    get_upscaler(stride=stride, output_size=output_size, order=order),
                    ConvNd(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
                ))
            else:
                raise ValueError('modules.conv: The arguments "stride" and "scaler" should be valid.')
            if layer_order == 'old':
                seq.append(
                    normalizer_op(out_planes)
                )
                new_activator = get_activator(activator, in_planes=out_planes)
                if new_activator is not None:
                    seq.append(
                        new_activator()
                    )
        else:
            raise ValueError('modules.conv: The arguments "normalizer"  should be valid.')
        self.mconv = nn.Sequential(*seq)

    def forward(self, x):
        x = self.mconv(x)
        return x


class _BlockConvStkNd(nn.Module):
    '''Create the N-D stacked modern convolutional block.
    Each block contains several stacked convolutional layers.
    When providing 'ex_planes', there would be a skip connection input
    for the first layer.
    This module is used for building UNetNd, AENd and Enc/DecoderNetNd,
    should not be used by users.
    '''
    def __init__(self, order, in_planes, out_planes, hidden_planes=None,
                 kernel_size=3, padding=1, stride=1, stack_level=3, ex_planes=0,
                 scaler='down', export_hidden=False):
        '''Initialization
        Arguments:
            order: the block dimension. For example, when order=2, the
                   nn.Conv2d would be used.
            in_planes: the channel number of the input data.
            out_planes: the channel number of the output data.
        Arguments (optional):
            hidden_planes: the channel number of the first hidden layer, would
                           also used as the base of the following channels. If
                           not set, would use "out_planes" as the default
                           value.
            kernel_size: the kernel size of the convolutional layers.
            padding: the padding size of the convolutional layers.
            stride: the stride size of the convolutional layers.
            stack_level: the number of convolutional layers in this block,
                         requiring to be >= 1.
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
            raise ValueError('modules.conv: The argument "stack_level" should be at least one.')

        # The first layer performs the scaling.
        self.with_exinput = isinstance(ex_planes, int) and ex_planes > 0
        self.export_hidden = export_hidden
        hidden_planes = hidden_planes if (isinstance(hidden_planes, int) and hidden_planes > 0) else out_planes
        self.conv_list = nn.ModuleList()
        for i in range(stack_level - 1):
            self.conv_list.append(
                _ConvModernNd(order, (in_planes + ex_planes) if i == 0 else hidden_planes, hidden_planes,
                              kernel_size=kernel_size, padding=padding, stride=1, scaler='down')
            )
        self.conv_scale = _ConvModernNd(order, hidden_planes if stack_level > 1 else (in_planes + ex_planes), out_planes,
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
    '''N-D convolutional network based U-Net
    This moule is a built-in model for convolutional U-Net. The network is
    inspired by:
        https://github.com/milesial/Pytorch-UNet
    The network would down-sample and up-sample the input data according to
    the network depth. The depth is given by the length of the argument
    "layers".
    '''
    def __init__(self, order, channel, layers, kernel_size=3, in_planes=1, out_planes=1):
        '''Initialization
        Arguments:
            order: the network dimension. For example, when order=2, the
                   nn.Conv2d would be used.
            channel: the channel number of the first layer, would also used
                     as the base of the following channels.
            layers: a list of layer numbers. Each number represents the number
                    of convolutional layers of a stage. The stage numer, i.e.
                    the depth of the network is the length of this list.
        Arguments (optional):
            kernel_size: the kernel size of each block.
            in_planes: the channel number of the input data.
            out_planes: the channel number of the output data.
        '''
        super().__init__()
        if len(layers) < 2:
            raise ValueError('modules.conv: The argument "layers" should contain at least 2 values, but provide "{0}"'.format(layers))
        self.__layers = layers
        ConvNd = get_convnd(order=order)

        ksize_e, psize_e, _ = cal_kernel_padding(kernel_size, ksize_plus=2)
        self.conv_first = ConvNd(in_planes, channel, kernel_size=ksize_e, stride=1, padding=psize_e, bias=False)
        self.conv_down_list = nn.ModuleList()
        self.conv_up_list = nn.ModuleList()
        ksize, psize, stride = cal_kernel_padding(kernel_size)

        # Down scaling route
        self.conv_down_list.append(
            _BlockConvStkNd(order, channel, channel, kernel_size=ksize, padding=psize,
                            stride=stride, stack_level=layers[0], ex_planes=0, scaler='down', export_hidden=True))
        for n_l in layers[1:-1]:
            self.conv_down_list.append(
                _BlockConvStkNd(order, channel, channel * 2, kernel_size=ksize, padding=psize,
                                stride=stride, stack_level=n_l, ex_planes=0, scaler='down', export_hidden=True))
            channel = channel * 2

        # Middle block
        self.conv_middle_up = _BlockConvStkNd(order, channel, channel, hidden_planes=channel * 2, kernel_size=ksize, padding=psize,
                                              stride=stride, stack_level=layers[-1], ex_planes=0, scaler='up')

        # Up scaling route
        for n_l in layers[-2:0:-1]:
            self.conv_up_list.append(
                _BlockConvStkNd(order, channel, channel // 2, hidden_planes=channel, kernel_size=ksize, padding=psize,
                                stride=stride, stack_level=n_l, ex_planes=channel, scaler='up'))
            channel = channel // 2
        self.conv_up_list.append(
            _BlockConvStkNd(order, channel, channel, hidden_planes=channel, kernel_size=ksize, padding=psize,
                            stride=1, stack_level=layers[0], ex_planes=channel, scaler='down'))
        self.conv_final = ConvNd(channel, out_planes, kernel_size=ksize_e, stride=1, padding=psize_e, bias=True)

    @property
    def nlayers(self):
        '''Return number of convolutional layers along the depth.
        '''
        if len(self.__layers) == 0:
            return 0
        n_layers = functools.reduce(lambda x, y: x + 2 * y, self.__layers[:-1], self.__layers[-1]) + 2
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
    '''N-D convolutional auto-encoder.
    This moule is a built-in model for convolutional auto-encoder.
    The network would down-sample and up-sample and the input data according
    to the network depth. The depth is given by the length of the argument
    "layers".
    '''
    def __init__(self, order, channel, layers, kernel_size=3, in_planes=1, out_planes=1):
        '''Initialization
        Arguments:
            order: the network dimension. For example, when order=2, the
                   nn.Conv2d would be used.
            channel: the channel number of the first layer, would also used
                     as the base of the following channels.
            layers: a list of layer numbers. Each number represents the number
                    of convolutional layers of a stage. The stage numer, i.e.
                    the depth of the network is the length of this list.
        Arguments (optional):
            kernel_size: the kernel size of each block.
            in_planes: the channel number of the input data.
            out_planes: the channel number of the output data.
        '''
        super().__init__()
        if len(layers) < 2:
            raise ValueError('modules.conv: The argument "layers" should contain at least 2 values, but provide "{0}"'.format(layers))
        self.__layers = layers
        ConvNd = get_convnd(order=order)

        ksize_e, psize_e, _ = cal_kernel_padding(kernel_size, ksize_plus=2)
        self.conv_first = ConvNd(in_planes, channel, kernel_size=ksize_e, stride=1, padding=psize_e, bias=False)
        self.conv_down_list = nn.ModuleList()
        self.conv_up_list = nn.ModuleList()
        ksize, psize, stride = cal_kernel_padding(kernel_size)

        # Down scaling route
        self.conv_down_list.append(
            _BlockConvStkNd(order, channel, channel, kernel_size=ksize, padding=psize,
                            stride=stride, stack_level=layers[0], ex_planes=0, scaler='down'))
        for n_l in layers[1:-1]:
            self.conv_down_list.append(
                _BlockConvStkNd(order, channel, channel * 2, kernel_size=ksize, padding=psize,
                                stride=stride, stack_level=n_l, ex_planes=0, scaler='down'))
            channel = channel * 2

        # Middle block
        self.conv_middle_up = _BlockConvStkNd(order, channel, channel, hidden_planes=channel * 2, kernel_size=ksize, padding=psize,
                                              stride=stride, stack_level=layers[-1], scaler='up')

        # Up scaling route
        for n_l in layers[-2:0:-1]:
            self.conv_up_list.append(
                _BlockConvStkNd(order, channel, channel // 2, hidden_planes=channel, kernel_size=ksize, padding=psize,
                                stride=stride, stack_level=n_l, scaler='up'))
            channel = channel // 2
        self.conv_up_list.append(
            _BlockConvStkNd(order, channel, channel, hidden_planes=channel, kernel_size=ksize, padding=psize,
                            stride=1, stack_level=layers[0], scaler='down'))
        self.conv_final = ConvNd(channel, out_planes, kernel_size=ksize_e, stride=1, padding=psize_e, bias=True)

    @property
    def nlayers(self):
        '''Return number of convolutional layers along the depth.
        '''
        if len(self.__layers) == 0:
            return 0
        n_layers = functools.reduce(lambda x, y: x + 2 * y, self.__layers[:-1], self.__layers[-1]) + 2
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
    '''N-D convolutional down-scale (encoder) network.
    This moule is a built-in model for convolutional network. The network
    could be used for down-scaling or classification.
    The network would down-sample and the input data according to the network
    depth. The depth is given by the length of the argument "layers".
    '''
    def __init__(self, order, channel, layers, kernel_size=3, in_planes=1, out_length=2):
        '''Initialization
        Arguments:
            order: the network dimension. For example, when order=2, the
                   nn.Conv2d would be used.
            channel: the channel number of the first layer, would also used
                     as the base of the following channels.
            layers: a list of layer numbers. Each number represents the number
                    of convolutional layers of a stage. The stage numer, i.e.
                    the depth of the network is the length of this list.
        Arguments (optional):
            kernel_size: the kernel size of each block.
            in_planes: the channel number of the input data.
            out_length: the length of the output vector, if not set, the
                        output would not be flattened.
        '''
        super().__init__()
        if len(layers) < 2:
            raise ValueError('modules.conv: The argument "layers" should contain at least 2 values, but provide "{0}"'.format(layers))
        self.__layers = layers
        ConvNd = get_convnd(order=order)

        ksize_e, psize_e, _ = cal_kernel_padding(kernel_size, ksize_plus=2)
        ksize, psize, stride = cal_kernel_padding(kernel_size)
        # Down scaling route
        netbody = nn.ModuleList([
            ConvNd(in_planes, channel, kernel_size=ksize_e, stride=1, padding=psize_e, bias=False),
            _BlockConvStkNd(order, channel, channel, kernel_size=ksize, padding=psize,
                            stride=stride, stack_level=layers[0], scaler='down')])
        for layer in layers[1:]:
            netbody.append(
                _BlockConvStkNd(order, channel, channel * 2, kernel_size=ksize, padding=psize,
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
        n_layers = functools.reduce(lambda x, y: x + y, self.__layers, 0) + 2
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
    '''N-D convolutional up-scale (decoder) network.
    This moule is a built-in model for convolutional network. The network
    could be used for up-scaling or generating samples.
    The network would up-sample and the input data according to the network
    depth. The depth is given by the length of the argument "layers".
    Different from the encoder network, this module requires the output
    shape, and the input shape is inferred from the given output shape.
    '''
    def __init__(self, order, channel, layers, out_size, kernel_size=3, in_length=2, out_planes=1):
        '''Initialization
        Arguments:
            order: the network dimension. For example, when order=2, the
                   nn.Conv2d would be used.
            channel: the channel number of the first layer, would also used
                     as the base of the following channels.
            layers: a list of layer numbers. Each number represents the number
                    of convolutional layers of a stage. The stage numer, i.e.
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
        super().__init__()
        if len(layers) < 2:
            raise ValueError('modules.conv: The argument "layers" should contain at least 2 values, but provide "{0}"'.format(layers))
        self.__layers = layers
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
                _BlockConvStkNd(order, channels[i], channels[i + 1], kernel_size=ksize, padding=psize,
                                stride=stride, stack_level=layer, scaler='up'))
        netbody.append(
            _BlockConvStkNd(order, channels[-1], channels[-1], kernel_size=ksize, padding=psize,
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
        n_layers = functools.reduce(lambda x, y: x + y, self.__layers, 0) + (3 if self.is_in_vector else 2)
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


class ConvModern1d(_ConvModernNd):
    '''1D Modern convolutional layer
    The implementation for the modern convolutional layer. The modern
    convolutional layer is a stack of convolution, normalization and
    activation.
    In the following paper, a new op composition order is proposed
    for building residual block:
        https://arxiv.org/abs/1603.05027
    We also support this implementation.
    '''
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, output_size=None,
                 normalizer='pinst', activator='prelu', layer_order='new', scaler='down'):
        '''Initialization.
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
        super().__init__(1, in_planes=in_planes, out_planes=out_planes,
                         kernel_size=kernel_size, stride=stride, padding=padding, output_size=output_size,
                         normalizer=normalizer, activator=activator,
                         layer_order=layer_order, scaler=scaler)


class ConvModern2d(_ConvModernNd):
    '''2D Modern convolutional layer
    The implementation for the modern convolutional layer. The modern
    convolutional layer is a stack of convolution, normalization and
    activation.
    In the following paper, a new op composition order is proposed
    for building residual block:
        https://arxiv.org/abs/1603.05027
    We also support this implementation.
    '''
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, output_size=None,
                 normalizer='pinst', activator='prelu', layer_order='new', scaler='down'):
        '''Initialization.
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
        super().__init__(2, in_planes=in_planes, out_planes=out_planes,
                         kernel_size=kernel_size, stride=stride, padding=padding, output_size=output_size,
                         normalizer=normalizer, activator=activator,
                         layer_order=layer_order, scaler=scaler)


class ConvModern3d(_ConvModernNd):
    '''3D Modern convolutional layer
    The implementation for the modern convolutional layer. The modern
    convolutional layer is a stack of convolution, normalization and
    activation.
    In the following paper, a new op composition order is proposed
    for building residual block:
        https://arxiv.org/abs/1603.05027
    We also support this implementation.
    '''
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, output_size=None,
                 normalizer='pinst', activator='prelu', layer_order='new', scaler='down'):
        '''Initialization.
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
        super().__init__(3, in_planes=in_planes, out_planes=out_planes,
                         kernel_size=kernel_size, stride=stride, padding=padding, output_size=output_size,
                         normalizer=normalizer, activator=activator,
                         layer_order=layer_order, scaler=scaler)


class UNet1d(_UNetNd):
    '''1D convolutional network based U-Net
    This moule is a built-in model for convolutional U-Net. The network is
    inspired by:
        https://github.com/milesial/Pytorch-UNet
    The network would down-sample and up-sample the input data according to
    the network depth. The depth is given by the length of the argument
    "layers".
    '''
    def __init__(self, channel, layers, kernel_size=3, in_planes=1, out_planes=1):
        '''Initialization
        Arguments:
            channel: the channel number of the first layer, would also used
                     as the base of the following channels.
            layers: a list of layer numbers. Each number represents the number
                    of convolutional layers of a stage. The stage numer, i.e.
                    the depth of the network is the length of this list.
        Arguments (optional):
            kernel_size: the kernel size of each block.
            in_planes: the channel number of the input data.
            out_planes: the channel number of the output data.
        '''
        super().__init__(1, channel=channel, layers=layers, kernel_size=kernel_size,
                         in_planes=in_planes, out_planes=out_planes)


class UNet2d(_UNetNd):
    '''2D convolutional network based U-Net
    This moule is a built-in model for convolutional U-Net. The network is
    inspired by:
        https://github.com/milesial/Pytorch-UNet
    The network would down-sample and up-sample the input data according to
    the network depth. The depth is given by the length of the argument
    "layers".
    '''
    def __init__(self, channel, layers, kernel_size=3, in_planes=1, out_planes=1):
        '''Initialization
        Arguments:
            channel: the channel number of the first layer, would also used
                     as the base of the following channels.
            layers: a list of layer numbers. Each number represents the number
                    of convolutional layers of a stage. The stage numer, i.e.
                    the depth of the network is the length of this list.
        Arguments (optional):
            kernel_size: the kernel size of each block.
            in_planes: the channel number of the input data.
            out_planes: the channel number of the output data.
        '''
        super().__init__(order=2, channel=channel, layers=layers, kernel_size=kernel_size,
                         in_planes=in_planes, out_planes=out_planes)


class UNet3d(_UNetNd):
    '''3D convolutional network based U-Net
    This moule is a built-in model for convolutional U-Net. The network is
    inspired by:
        https://github.com/milesial/Pytorch-UNet
    The network would down-sample and up-sample the input data according to
    the network depth. The depth is given by the length of the argument
    "layers".
    '''
    def __init__(self, channel, layers, kernel_size=3, in_planes=1, out_planes=1):
        '''Initialization
        Arguments:
            channel: the channel number of the first layer, would also used
                     as the base of the following channels.
            layers: a list of layer numbers. Each number represents the number
                    of convolutional layers of a stage. The stage numer, i.e.
                    the depth of the network is the length of this list.
        Arguments (optional):
            kernel_size: the kernel size of each block.
            in_planes: the channel number of the input data.
            out_planes: the channel number of the output data.
        '''
        super().__init__(order=3, channel=channel, layers=layers, kernel_size=kernel_size,
                         in_planes=in_planes, out_planes=out_planes)


class AE1d(_AENd):
    '''1D convolutional auto-encoder.
    This moule is a built-in model for convolutional auto-encoder.
    The network would down-sample and up-sample and the input data according
    to the network depth. The depth is given by the length of the argument
    "layers".
    '''
    def __init__(self, channel, layers, kernel_size=3, in_planes=1, out_planes=1):
        '''Initialization
        Arguments:
            channel: the channel number of the first layer, would also used
                     as the base of the following channels.
            layers: a list of layer numbers. Each number represents the number
                    of convolutional layers of a stage. The stage numer, i.e.
                    the depth of the network is the length of this list.
        Arguments (optional):
            kernel_size: the kernel size of each block.
            in_planes: the channel number of the input data.
            out_planes: the channel number of the output data.
        '''
        super().__init__(1, channel=channel, layers=layers, kernel_size=kernel_size,
                         in_planes=in_planes, out_planes=out_planes)


class AE2d(_AENd):
    '''2D convolutional auto-encoder.
    This moule is a built-in model for convolutional auto-encoder.
    The network would down-sample and up-sample and the input data according
    to the network depth. The depth is given by the length of the argument
    "layers".
    '''
    def __init__(self, channel, layers, kernel_size=3, in_planes=1, out_planes=1):
        '''Initialization
        Arguments:
            channel: the channel number of the first layer, would also used
                     as the base of the following channels.
            layers: a list of layer numbers. Each number represents the number
                    of convolutional layers of a stage. The stage numer, i.e.
                    the depth of the network is the length of this list.
        Arguments (optional):
            kernel_size: the kernel size of each block.
            in_planes: the channel number of the input data.
            out_planes: the channel number of the output data.
        '''
        super().__init__(2, channel=channel, layers=layers, kernel_size=kernel_size,
                         in_planes=in_planes, out_planes=out_planes)


class AE3d(_AENd):
    '''3D convolutional auto-encoder.
    This moule is a built-in model for convolutional auto-encoder.
    The network would down-sample and up-sample and the input data according
    to the network depth. The depth is given by the length of the argument
    "layers".
    '''
    def __init__(self, channel, layers, kernel_size=3, in_planes=1, out_planes=1):
        '''Initialization
        Arguments:
            channel: the channel number of the first layer, would also used
                     as the base of the following channels.
            layers: a list of layer numbers. Each number represents the number
                    of convolutional layers of a stage. The stage numer, i.e.
                    the depth of the network is the length of this list.
        Arguments (optional):
            kernel_size: the kernel size of each block.
            in_planes: the channel number of the input data.
            out_planes: the channel number of the output data.
        '''
        super().__init__(3, channel=channel, layers=layers, kernel_size=kernel_size,
                         in_planes=in_planes, out_planes=out_planes)


class EncoderNet1d(_EncoderNetNd):
    '''1D convolutional down-scale (encoder) network.
    This moule is a built-in model for convolutional network. The network
    could be used for down-scaling or classification.
    The network would down-sample and the input data according to the network
    depth. The depth is given by the length of the argument "layers".
    '''
    def __init__(self, channel, layers, kernel_size=3, in_planes=1, out_length=2):
        '''Initialization
        Arguments:
            channel: the channel number of the first layer, would also used
                     as the base of the following channels.
            layers: a list of layer numbers. Each number represents the number
                    of convolutional layers of a stage. The stage numer, i.e.
                    the depth of the network is the length of this list.
        Arguments (optional):
            kernel_size: the kernel size of each block.
            in_planes: the channel number of the input data.
            out_length: the length of the output vector, if not set, the
                        output would not be flattened.
        '''
        super().__init__(1, channel=channel, layers=layers, kernel_size=kernel_size,
                         in_planes=in_planes, out_length=out_length)


class EncoderNet2d(_EncoderNetNd):
    '''2D convolutional down-scale (encoder) network.
    This moule is a built-in model for convolutional network. The network
    could be used for down-scaling or classification.
    The network would down-sample and the input data according to the network
    depth. The depth is given by the length of the argument "layers".
    '''
    def __init__(self, channel, layers, kernel_size=3, in_planes=1, out_length=2):
        '''Initialization
        Arguments:
            channel: the channel number of the first layer, would also used
                     as the base of the following channels.
            layers: a list of layer numbers. Each number represents the number
                    of convolutional layers of a stage. The stage numer, i.e.
                    the depth of the network is the length of this list.
        Arguments (optional):
            kernel_size: the kernel size of each block.
            in_planes: the channel number of the input data.
            out_length: the length of the output vector, if not set, the
                        output would not be flattened.
        '''
        super().__init__(2, channel=channel, layers=layers, kernel_size=kernel_size,
                         in_planes=in_planes, out_length=out_length)


class EncoderNet3d(_EncoderNetNd):
    '''3D convolutional down-scale (encoder) network.
    This moule is a built-in model for convolutional network. The network
    could be used for down-scaling or classification.
    The network would down-sample and the input data according to the network
    depth. The depth is given by the length of the argument "layers".
    '''
    def __init__(self, channel, layers, kernel_size=3, in_planes=1, out_length=2):
        '''Initialization
        Arguments:
            channel: the channel number of the first layer, would also used
                     as the base of the following channels.
            layers: a list of layer numbers. Each number represents the number
                    of convolutional layers of a stage. The stage numer, i.e.
                    the depth of the network is the length of this list.
        Arguments (optional):
            kernel_size: the kernel size of each block.
            in_planes: the channel number of the input data.
            out_length: the length of the output vector, if not set, the
                        output would not be flattened.
        '''
        super().__init__(3, channel=channel, layers=layers, kernel_size=kernel_size,
                         in_planes=in_planes, out_length=out_length)


class DecoderNet1d(_DecoderNetNd):
    '''1D convolutional up-scale (decoder) network.
    This moule is a built-in model for convolutional network. The network
    could be used for up-scaling or generating samples.
    The network would up-sample and the input data according to the network
    depth. The depth is given by the length of the argument "layers".
    Different from the encoder network, this module requires the output
    shape, and the input shape is inferred from the given output shape.
    '''
    def __init__(self, channel, layers, out_size, kernel_size=3, in_length=2, out_planes=1):
        '''Initialization
        Arguments:
            channel: the channel number of the first layer, would also used
                     as the base of the following channels.
            layers: a list of layer numbers. Each number represents the number
                    of convolutional layers of a stage. The stage numer, i.e.
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
        super().__init__(1, channel=channel, layers=layers, out_size=out_size,
                         kernel_size=kernel_size, in_length=in_length, out_planes=out_planes)


class DecoderNet2d(_DecoderNetNd):
    '''2D convolutional up-scale (decoder) network.
    This moule is a built-in model for convolutional network. The network
    could be used for up-scaling or generating samples.
    The network would up-sample and the input data according to the network
    depth. The depth is given by the length of the argument "layers".
    Different from the encoder network, this module requires the output
    shape, and the input shape is inferred from the given output shape.
    '''
    def __init__(self, channel, layers, out_size, kernel_size=3, in_length=2, out_planes=1):
        '''Initialization
        Arguments:
            channel: the channel number of the first layer, would also used
                     as the base of the following channels.
            layers: a list of layer numbers. Each number represents the number
                    of convolutional layers of a stage. The stage numer, i.e.
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
        super().__init__(2, channel=channel, layers=layers, out_size=out_size,
                         kernel_size=kernel_size, in_length=in_length, out_planes=out_planes)


class DecoderNet3d(_DecoderNetNd):
    '''3D convolutional up-scale (decoder) network.
    This moule is a built-in model for convolutional network. The network
    could be used for up-scaling or generating samples.
    The network would up-sample and the input data according to the network
    depth. The depth is given by the length of the argument "layers".
    Different from the encoder network, this module requires the output
    shape, and the input shape is inferred from the given output shape.
    '''
    def __init__(self, channel, layers, out_size, kernel_size=3, in_length=2, out_planes=1):
        '''Initialization
        Arguments:
            channel: the channel number of the first layer, would also used
                     as the base of the following channels.
            layers: a list of layer numbers. Each number represents the number
                    of convolutional layers of a stage. The stage numer, i.e.
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
        super().__init__(3, channel=channel, layers=layers, out_size=out_size,
                         kernel_size=kernel_size, in_length=in_length, out_planes=out_planes)


def __get_unet_nd(order):
    if order == 3:
        return UNet3d
    elif order == 2:
        return UNet2d
    elif order == 1:
        return UNet1d
    else:
        raise ValueError('modules.conv: The argument "order" could only be 1, 2, or 3.')


def __get_ae_nd(order):
    if order == 3:
        return AE3d
    elif order == 2:
        return AE2d
    elif order == 1:
        return AE1d
    else:
        raise ValueError('modules.conv: The argument "order" could only be 1, 2, or 3.')


def __get_encnet_nd(order):
    if order == 3:
        return EncoderNet3d
    elif order == 2:
        return EncoderNet2d
    elif order == 1:
        return EncoderNet1d
    else:
        raise ValueError('modules.conv: The argument "order" could only be 1, 2, or 3.')


def __get_decnet_nd(order):
    if order == 3:
        return DecoderNet3d
    elif order == 2:
        return DecoderNet2d
    elif order == 1:
        return DecoderNet1d
    else:
        raise ValueError('modules.conv: The argument "order" could only be 1, 2, or 3.')


# unet
def unet12(order=2, **kwargs):
    '''Constructs a conv.UNet-12 model.
    Configurations:
        Network depth: 3
        Stage details: [2, 2, 2]
        First channel number: 64
    Arguments:
        order: the dimension of the network. For example, when
               order=2, the nn.Conv2d would be used.
    Other Arguments (see mdnc.modules.conv.UNet*d):
        in_planes, out_planes, kernel_size
    '''
    model = __get_unet_nd(order)(64, [2, 2, 2], **kwargs)
    return model


def unet16(order=2, **kwargs):
    '''Constructs a conv.UNet-16 model.
    Configurations:
        Network depth: 4
        Stage details: [2, 2, 2, 2]
        First channel number: 64
    Arguments:
        order: the dimension of the network. For example, when
               order=2, the nn.Conv2d would be used.
    Other Arguments (see mdnc.modules.conv.UNet*d):
        in_planes, out_planes, kernel_size
    '''
    model = __get_unet_nd(order)(64, [2, 2, 2, 2], **kwargs)
    return model


def unet17(order=2, **kwargs):
    '''Constructs a conv.UNet-17 model.
    Configurations:
        Network depth: 3
        Stage details: [3, 3, 3]
        First channel number: 64
    Arguments:
        order: the dimension of the network. For example, when
               order=2, the nn.Conv2d would be used.
    Other Arguments (see mdnc.modules.conv.UNet*d):
        in_planes, out_planes, kernel_size
    '''
    model = __get_unet_nd(order)(64, [3, 3, 3], **kwargs)
    return model


def unet23(order=2, **kwargs):
    '''Constructs a conv.UNet-23 model.
    Configurations:
        Network depth: 4
        Stage details: [3, 3, 3, 3]
        First channel number: 64
    Arguments:
        order: the dimension of the network. For example, when
               order=2, the nn.Conv2d would be used.
    Other Arguments (see mdnc.modules.conv.UNet*d):
        in_planes, out_planes, kernel_size
    '''
    model = __get_unet_nd(order)(64, [3, 3, 3, 3], **kwargs)
    return model


def unet29(order=2, **kwargs):
    '''Constructs a conv.UNet-29 model.
    This model is equivalent to
        https://github.com/milesial/Pytorch-UNet
    Configurations:
        Network depth: 5
        Stage details: [3, 3, 3, 3, 3]
        First channel number: 64
    Arguments:
        order: the dimension of the network. For example, when
               order=2, the nn.Conv2d would be used.
    Other Arguments (see mdnc.modules.conv.UNet*d):
        in_planes, out_planes, kernel_size
    '''
    model = __get_unet_nd(order)(64, [3, 3, 3, 3, 3], **kwargs)
    return model


# ae
def ae12(order=2, **kwargs):
    '''Constructs a conv.AE-12 model.
    Configurations:
        Network depth: 3
        Stage details: [2, 2, 2]
        First channel number: 64
    Arguments:
        order: the dimension of the network. For example, when
               order=2, the nn.Conv2d would be used.
    Other Arguments (see mdnc.modules.conv.AE*d):
        in_planes, out_planes, kernel_size
    '''
    model = __get_ae_nd(order)(64, [2, 2, 2], **kwargs)
    return model


def ae16(order=2, **kwargs):
    '''Constructs a conv.AE-16 model.
    Configurations:
        Network depth: 4
        Stage details: [2, 2, 2, 2]
        First channel number: 64
    Arguments:
        order: the dimension of the network. For example, when
               order=2, the nn.Conv2d would be used.
    Other Arguments (see mdnc.modules.conv.AE*d):
        in_planes, out_planes, kernel_size
    '''
    model = __get_ae_nd(order)(64, [2, 2, 2, 2], **kwargs)
    return model


def ae17(order=2, **kwargs):
    '''Constructs a conv.AE-17 model.
    Configurations:
        Network depth: 3
        Stage details: [3, 3, 3]
        First channel number: 64
    Arguments:
        order: the dimension of the network. For example, when
               order=2, the nn.Conv2d would be used.
    Other Arguments (see mdnc.modules.conv.AE*d):
        in_planes, out_planes, kernel_size
    '''
    model = __get_ae_nd(order)(64, [3, 3, 3], **kwargs)
    return model


def ae23(order=2, **kwargs):
    '''Constructs a conv.AE-23 model.
    Configurations:
        Network depth: 4
        Stage details: [3, 3, 3, 3]
        First channel number: 64
    Arguments:
        order: the dimension of the network. For example, when
               order=2, the nn.Conv2d would be used.
    Other Arguments (see mdnc.modules.conv.AE*d):
        in_planes, out_planes, kernel_size
    '''
    model = __get_ae_nd(order)(64, [3, 3, 3, 3], **kwargs)
    return model


def ae29(order=2, **kwargs):
    '''Constructs a conv.AE-29 model.
    Configurations:
        Network depth: 5
        Stage details: [3, 3, 3, 3, 3]
        First channel number: 64
    Arguments:
        order: the dimension of the network. For example, when
               order=2, the nn.Conv2d would be used.
    Other Arguments (see mdnc.modules.conv.AE*d):
        in_planes, out_planes, kernel_size
    '''
    model = __get_ae_nd(order)(64, [3, 3, 3, 3, 3], **kwargs)
    return model


# encnet
def encnet12(order=2, **kwargs):
    '''Constructs a conv.EncoderNet-12 model.
    Configurations:
        Network depth: 5
        Stage details: [2, 2, 2, 2, 2]
        First channel number: 64
    Arguments:
        order: the dimension of the network. For example, when
               order=2, the nn.Conv2d would be used.
    Other Arguments (see mdnc.modules.conv.EncoderNet*d):
        in_planes, out_length, kernel_size
    '''
    model = __get_encnet_nd(order)(64, [2, 2, 2, 2, 2], **kwargs)
    return model


def encnet15(order=2, **kwargs):
    '''Constructs a conv.EncoderNet-15 model.
    Configurations:
        Network depth: 5
        Stage details: [2, 2, 3, 3, 3]
        First channel number: 64
    Arguments:
        order: the dimension of the network. For example, when
               order=2, the nn.Conv2d would be used.
    Other Arguments (see mdnc.modules.conv.EncoderNet*d):
        in_planes, out_length, kernel_size
    '''
    model = __get_encnet_nd(order)(64, [2, 2, 3, 3, 3], **kwargs)
    return model


def encnet17(order=2, **kwargs):
    '''Constructs a conv.EncoderNet-17 model.
    Configurations:
        Network depth: 5
        Stage details: [3, 3, 3, 3, 3]
        First channel number: 64
    Arguments:
        order: the dimension of the network. For example, when
               order=2, the nn.Conv2d would be used.
    Other Arguments (see mdnc.modules.conv.EncoderNet*d):
        in_planes, out_length, kernel_size
    '''
    model = __get_encnet_nd(order)(64, [3, 3, 3, 3, 3], **kwargs)
    return model


def encnet22(order=2, **kwargs):
    '''Constructs a conv.EncoderNet-22 model.
    Configurations:
        Network depth: 5
        Stage details: [4, 4, 4, 4, 4]
        First channel number: 64
    Arguments:
        order: the dimension of the network. For example, when
               order=2, the nn.Conv2d would be used.
    Other Arguments (see mdnc.modules.conv.EncoderNet*d):
        in_planes, out_length, kernel_size
    '''
    model = __get_encnet_nd(order)(64, [4, 4, 4, 4, 4], **kwargs)
    return model


# decnet
def decnet13(out_size, order=2, **kwargs):
    '''Constructs a conv.DecoderNet-13 model.
    Configurations:
        Network depth: 5
        Stage details: [2, 2, 2, 2, 2]
        First channel number: 64
    Arguments:
        out_size: the output shape of the network.
        order: the dimension of the network. For example, when
               order=2, the nn.Conv2d would be used.
    Other Arguments (see mdnc.modules.conv.DecoderNet*d):
        in_length, out_planes, kernel_size
    '''
    model = __get_decnet_nd(order)(64, [2, 2, 2, 2, 2], out_size=out_size, **kwargs)
    return model


def decnet16(out_size, order=2, **kwargs):
    '''Constructs a conv.DecoderNet-16 model.
    Configurations:
        Network depth: 5
        Stage details: [3, 3, 2, 2, 2]
        First channel number: 64
    Arguments:
        out_size: the output shape of the network.
        order: the dimension of the network. For example, when
               order=2, the nn.Conv2d would be used.
    Other Arguments (see mdnc.modules.conv.DecoderNet*d):
        in_length, out_planes, kernel_size
    '''
    model = __get_decnet_nd(order)(64, [3, 3, 2, 2, 2], out_size=out_size, **kwargs)
    return model


def decnet18(out_size, order=2, **kwargs):
    '''Constructs a conv.DecoderNet-18 model.
    Configurations:
        Network depth: 5
        Stage details: [3, 3, 3, 3, 3]
        First channel number: 64
    Arguments:
        out_size: the output shape of the network.
        order: the dimension of the network. For example, when
               order=2, the nn.Conv2d would be used.
    Other Arguments (see mdnc.modules.conv.DecoderNet*d):
        in_length, out_planes, kernel_size
    '''
    model = __get_decnet_nd(order)(64, [3, 3, 3, 3, 3], out_size=out_size, **kwargs)
    return model


def decnet23(out_size, order=2, **kwargs):
    '''Constructs a conv.DecoderNet-23 model.
    Configurations:
        Network depth: 5
        Stage details: [4, 4, 4, 4, 4]
        First channel number: 64
    Arguments:
        out_size: the output shape of the network.
        order: the dimension of the network. For example, when
               order=2, the nn.Conv2d would be used.
    Other Arguments (see mdnc.modules.conv.DecoderNet*d):
        in_length, out_planes, kernel_size
    '''
    model = __get_decnet_nd(order)(64, [4, 4, 4, 4, 4], out_size=out_size, **kwargs)
    return model
