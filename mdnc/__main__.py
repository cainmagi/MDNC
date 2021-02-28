#!python
# -*- coding: UTF-8 -*-
'''
################################################################
# Compatibility tests
# @ Modern Deep Network Toolkits for pyTorch
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.5+
#   pyTorch 1.0.0+
#   numpy 1.13+
#   matplotlib 3.1.1+
#   ... other modules, see setup.py
# The entry code for all compatibility tests.
################################################################
'''

import collections
import argparse

from mdnc import __version__
from mdnc.contribs import __main__ as test_contribs
from mdnc.data import __main__ as test_data
from mdnc.modules import __main__ as test_modules
from mdnc.utils import __main__ as test_utils


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
        '-all', '--test_all', type=str2bool, nargs='?', const=True, default=False, metavar='bool',
        help='''Test all sub-modules.'''
    )
    if return_args:
        return parser.parse_args()
    else:
        return


if __name__ == '__main__':
    __spec__ = None  # Handle the error caused by pdb module.

    print('Compatibility test: mdnc.utils. MDNC version: ', __version__)

    # Set parser and parse args.
    aparser = argparse.ArgumentParser(
        description='Compatibility test: mdnc.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    test_contribs.parse_args(aparser, return_args=False)
    test_data.parse_args(aparser, return_args=False)
    test_modules.parse_args(aparser, return_args=False)
    test_utils.parse_args(aparser, return_args=False)
    args = vars(parse_args(aparser))

    registered_tests = collections.OrderedDict()
    registered_tests.update(test_contribs.registered_tests)
    registered_tests.update(test_data.registered_tests)
    registered_tests.update(test_modules.registered_tests)
    registered_tests.update(test_utils.registered_tests)

    if not any(args.values()):
        aparser.print_help()
        is_all = input('No test specified explicitly. Do you want to run all tests? (y/n)')
        is_all = str2bool(is_all)
        if is_all:
            for tfunc in registered_tests.values():
                tfunc()
    elif args['test_all']:
        for tfunc in registered_tests.values():
            tfunc()
    else:
        for k, req_run in args.items():
            if req_run:
                registered_tests[k]()
