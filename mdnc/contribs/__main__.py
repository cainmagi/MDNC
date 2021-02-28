#!python
# -*- coding: UTF-8 -*-
'''
################################################################
# Contribs - Compatibility tests
# @ Modern Deep Network Toolkits for pyTorch
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.5+
#   pyTorch 1.0.0+
# Run the following command to perform the test:
# ```bash
# python -m mdnc.contribs
# ```
################################################################
'''

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from mdnc import __version__
import mdnc.contribs as engine


class _TestTupleOutModule(nn.Module):
    '''Test module for torchsummary.
    This module is modified from here:
        https://github.com/sksq96/pytorch-summary/blob/011b2bd0ec7153d5842c1b37d1944fc6a7bf5feb/torchsummary/tests/test_models/test_model.py#L39
    '''
    def __init__(self):
        super().__init__()
        self.fc1a = nn.Linear(300, 50)
        self.fc1b = nn.Linear(50, 10)

        self.fc2a = nn.Linear(300, 50)
        self.fc2b = nn.Linear(50, 10)

    def forward(self, x1, x2):
        x1 = F.relu(self.fc1a(x1))
        x1 = self.fc1b(x1)
        x2 = x2.type(torch.FloatTensor)
        x2 = F.relu(self.fc2a(x2))
        x2 = self.fc2b(x2)
        # set x2 to FloatTensor
        x = torch.cat((x1, x2), 0)
        return F.log_softmax(x, dim=1), F.log_softmax(x1, dim=1), F.log_softmax(x2, dim=1)


class _TestDictOutModule(nn.Module):
    '''Test module for torchsummary.
    See the module here:
        https://github.com/sksq96/pytorch-summary/blob/011b2bd0ec7153d5842c1b37d1944fc6a7bf5feb/torchsummary/tests/test_models/test_model.py#L39
    '''
    def __init__(self):
        super().__init__()
        self.fc1a = nn.Linear(300, 50)
        self.fc1b = nn.Linear(50, 10)

        self.fc2 = nn.Sequential(
            nn.Linear(300, 50),
            nn.ReLU(),
            nn.Linear(50, 10))

    def forward(self, x1, x2):
        x1 = F.relu(self.fc1a(x1))
        x1 = self.fc1b(x1)
        x2 = x2.type(torch.FloatTensor)
        x2 = self.fc2(x2)
        # set x2 to FloatTensor
        x = torch.cat((x1, x2), 0)
        return {
            'x': F.log_softmax(x, dim=1),
            'x1': F.log_softmax(x1, dim=1),
            'x2': F.log_softmax(x2, dim=1)
        }


class TestTorchSummary:
    '''Test functions for torchsummary sub-module.
    '''
    def __init__(self):
        self.module_tuple = _TestTupleOutModule()
        self.module_dict = _TestDictOutModule()

    def test(self):
        '''See
        https://github.com/sksq96/pytorch-summary/blob/011b2bd0ec7153d5842c1b37d1944fc6a7bf5feb/torchsummary/tests/unit_tests/torchsummary_test.py#L41
        '''
        input1 = (1, 300)
        input2 = (1, 300)
        dtypes = (torch.FloatTensor, torch.LongTensor)
        total_params, trainable_params = engine.torchsummary.summary(
            self.module_tuple, (input1, input2), device='cpu', dtypes=dtypes)
        total_params, trainable_params = engine.torchsummary.summary(
            self.module_dict, (input1, input2), device='cpu', dtypes=dtypes)


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
        '-ctcs', '--test_con_torchsummary', type=str2bool, nargs='?', const=True, default=False, metavar='bool',
        help='''Test the contribs.torchsummary module.'''
    )
    if return_args:
        return parser.parse_args()
    else:
        return


# Test functions
def test_con_torchsummary():
    print('Compatibility test: mdnc.contribs.torchsummary.')
    tester = TestTorchSummary()
    tester.test()


registered_tests = {
    'test_con_torchsummary': test_con_torchsummary
}


if __name__ == '__main__':
    __spec__ = None  # Handle the error caused by pdb module.

    print('Compatibility test: mdnc.contribs. MDNC version: ', __version__)

    # Set parser and parse args.
    aparser = argparse.ArgumentParser(
        description='Compatibility test: mdnc.contribs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    args = vars(parse_args(aparser))

    if not any(args.values()):
        aparser.print_help()
    else:
        for k, req_run in args.items():
            if req_run:
                registered_tests[k]()
