#!python
# -*- coding: UTF-8 -*-
'''
################################################################
# Modules - Compatibility tests
# @ Modern Deep Network Toolkits for pyTorch
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.5+
#   pyTorch 1.0.0+
# Run the following command to perform the test:
# ```bash
# python -m mdnc.modules
# ```
################################################################
'''

import abc
import argparse

import torch.nn as nn

from mdnc import __version__
from mdnc.contribs import torchsummary
import mdnc.modules as engine


class TestModuleAbstract(abc.ABC):
    '''Abstract test functions for sub-modules.
    '''
    @abc.abstractmethod
    def __init__(self):
        '''Initialization
        Neet to implement the following paramters.
            self.layers, self.networks
        '''
        self.layers_1d = list()
        self.layers_2d = list()
        self.layers_3d = list()
        self.networks = list()
        self.net_decs = list()

        self.orders = (1, 2, 3)
        self.input_sizes = (
            (3, 625),
            (3, 125, 125),
            (3, 25, 25, 25)
        )

    def test_layers(self):
        layers = (
            self.layers_1d,
            self.layers_2d,
            self.layers_3d
        )
        out_planes = 2
        for order, input_size, layers in zip(self.orders, self.input_sizes, layers):
            print('modules.modules: Test {0}d layers.'.format(order))
            for layer in layers:
                test_module = nn.Sequential(
                    layer(input_size[0], out_planes, kernel_size=3, stride=2, padding=1, scaler='down'),
                    layer(out_planes, input_size[0], kernel_size=3, stride=2, padding=1, output_size=input_size[1:], scaler='up')
                )
                torchsummary.summary(test_module, input_size=input_size, device='cpu')
                del test_module

    def test_networks(self):
        for order, input_size in zip(self.orders, self.input_sizes):
            print('modules.modules: Test {0}d networks.'.format(order))
            for net in self.networks:
                test_module = net(order=order, in_planes=input_size[0])
                print('{0} with {1} layers along its depth.'.format(type(test_module).__name__, test_module.nlayers))
                torchsummary.summary(test_module, input_size=input_size, device='cpu')
                del test_module

    def test_decodernets(self):
        for order, out_size in zip(self.orders, self.input_sizes):
            print('modules.modules: Test {0}d decoders.'.format(order))
            for net in self.net_decs:
                test_module = net(order=order, in_length=2, out_size=out_size[1:])
                print('{0} with {1} layers along its depth.'.format(type(test_module).__name__, test_module.nlayers))
                torchsummary.summary(test_module, input_size=(2, ), device='cpu')
                del test_module


class TestConv(TestModuleAbstract):
    '''Test functions for conv sub-module.
    '''
    def __init__(self):
        super().__init__()
        self.layers_1d = [engine.conv.ConvModern1d, ]
        self.layers_2d = [engine.conv.ConvModern2d, ]
        self.layers_3d = [engine.conv.ConvModern3d, ]
        self.networks = [engine.conv.unet29, engine.conv.ae29, engine.conv.encnet22]
        self.net_decs = [engine.conv.decnet23, ]


class TestResNet(TestModuleAbstract):
    '''Test functions for resnet sub-module.
    '''
    def __init__(self):
        super().__init__()
        self.layers_1d = [engine.resnet.BlockPlain1d, engine.resnet.BlockBottleneck1d, ]
        self.layers_2d = [engine.resnet.BlockPlain2d, engine.resnet.BlockBottleneck2d, ]
        self.layers_3d = [engine.resnet.BlockPlain3d, engine.resnet.BlockBottleneck3d, ]
        self.networks = [engine.resnet.unet83, engine.resnet.ae83, engine.resnet.encnet62]
        self.net_decs = [engine.resnet.decnet63, ]


# Argparser
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args(parser, return_args=True):
    parser.add_argument(
        '-mcon', '--test_mod_conv', type=str2bool, nargs='?', const=True, default=False, metavar='bool',
        help='''Test the modules.conv module.'''
    )
    parser.add_argument(
        '-mres', '--test_mod_resnet', type=str2bool, nargs='?', const=True, default=False, metavar='bool',
        help='''Test the modules.resnet module.'''
    )
    if return_args:
        return parser.parse_args()
    else:
        return


# Test functions
def test_mod_conv():
    print('Compatibility test: mdnc.modules.conv.')
    tester = TestConv()
    tester.test_layers()
    tester.test_networks()
    tester.test_decodernets()


def test_mod_resnet():
    print('Compatibility test: mdnc.modules.resnet.')
    tester = TestResNet()
    tester.test_layers()
    tester.test_networks()
    tester.test_decodernets()


registered_tests = {
    'test_mod_conv': test_mod_conv,
    'test_mod_resnet': test_mod_resnet
}


if __name__ == '__main__':
    __spec__ = None  # Handle the error caused by pdb module.

    print('Compatibility test: mdnc.utils. MDNC version: ', __version__)

    # Set parser and parse args.
    aparser = argparse.ArgumentParser(
        description='Compatibility test: mdnc.utils.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    args = vars(parse_args(aparser))

    if not any(args.values()):
        aparser.print_help()
    else:
        for k, req_run in args.items():
            if req_run:
                registered_tests[k]()
