#!python
# -*- coding: UTF-8 -*-
'''
################################################################
# Utilities - Compatibility tests
# @ Modern Deep Network Toolkits for pyTorch
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.5+
#   numpy 1.13+
#   matplotlib 3.1.1+
# Run the following command to perform the test:
# ```bash
# python -m mdnc.utils
# ```
################################################################
'''

import time
import argparse

import numpy as np
import matplotlib.pyplot as plt
import tqdm

from mdnc import __version__
import mdnc.utils as engine


class TestDraw:
    '''Test functions for draw sub-module.
    '''
    def test_all(self):
        self.test_plot_hist()
        self.test_plot_bar()
        self.test_scatter()
        self.test_training_records()
        self.test_error_bar()
        self.test_distribution()

    @engine.draw.setFigure(style='ggplot', font_size=14)
    def test_plot_hist(self):
        def func_gen():
            getbins = np.linspace(0, 25, 80)
            x1 = np.random.normal(loc=7.0, scale=1.0, size=100)
            yield x1, {'bins': getbins, 'label': '$x_1$'}
            x2 = np.random.normal(loc=12.0, scale=3.0, size=1000)
            yield x2, {'bins': getbins, 'label': '$x_2$'}
        engine.draw.plot_hist(func_gen(), xlabel='x', normalized=True, cumulative=False)
        engine.draw.plt.show()

    @engine.draw.setFigure(style='dark_background', font_size=14)
    def test_plot_bar(self):
        def func_gen():
            size = 5
            x1 = np.abs(np.random.normal(loc=6.0, scale=3.0, size=size))
            yield x1, {'label': '$x_1$'}
            x2 = np.abs(np.random.normal(loc=9.0, scale=6.0, size=size))
            yield x2, {'label': '$x_2$'}
            x3 = np.abs(np.random.normal(loc=12.0, scale=6.0, size=size))
            yield x3, {'label': '$x_3$'}
        engine.draw.plot_bar(func_gen(), num=3, ylabel='y', y_log=False,
                             x_tick_labels=['Jan.', 'Feb.', 'Mar.', 'Apr.', 'May'])
        engine.draw.plt.show()

    @engine.draw.setFigure(style='seaborn-darkgrid', font_size=16)
    def test_scatter(self):
        def func_gen():
            size = 100
            for i in range(3):
                center = -4.0 + 4.0 * np.random.rand(2)
                scale = 0.5 + 2.0 * np.random.rand(2)
                x1 = np.random.normal(loc=center[0], scale=scale[0], size=size)
                x2 = np.random.normal(loc=center[1], scale=scale[1], size=size)
                yield np.power(10, x1), np.power(10, x2), {'label': r'$x_{' + str(i + 1) + r'}$'}
        engine.draw.plot_scatter(func_gen(), x_log=True, y_log=True,
                                 xlabel='Metric 1', ylabel='Metric 2')
        engine.draw.plt.show()

    @engine.draw.setFigure(style='Solarize_Light2', font_size=14)
    def test_training_records(self):
        def func_gen_batch():
            size = 100
            x = np.arange(start=0, stop=size)
            for i in range(3):
                begin = 1 + 99.0 * np.random.rand()
                end = 2 + 10 * np.random.rand()
                v = begin * np.exp((np.square((x - size) / size) - 1.0) * end)
                yield x, v, {'label': r'$x_{' + str(i + 1) + r'}$'}

        def func_gen_epoch():
            size = 10
            x = np.arange(start=0, stop=size)
            for i in range(3):
                begin = 1 + 99.0 * np.random.rand()
                end = 2 + 10 * np.random.rand()
                v = begin * np.exp((np.square((x - size) / size) - 1.0) * end)
                val_v = begin * np.exp((np.square((x - size) / size) - 1.0) * (end - 1))
                data = np.stack([x, v, x, val_v], axis=0)
                yield data, {'label': r'$x_{' + str(i + 1) + r'}$'}

        engine.draw.plot_training_records(func_gen_batch(), y_log=True, x_mark_num=10,
                                          xlabel='Step', ylabel=r'Batch $\mathcal{L}$')
        engine.draw.plt.show()
        engine.draw.plot_training_records(func_gen_epoch(), y_log=True, x_mark_num=10,
                                          xlabel='Step', ylabel=r'Epoch $\mathcal{L}$')
        engine.draw.plt.show()

    @engine.draw.setFigure(style='bmh', font_size=16)
    def test_error_bar(self):
        def func_gen():
            size = 100
            x = np.arange(start=0, stop=size)
            for i in range(3):
                begin = 1 + 99.0 * np.random.rand()
                end = 2 + 10 * np.random.rand()
                exp_v = np.square((x - size) / size) - 1.0
                exp_vnoise = np.random.normal(0.0, np.expand_dims((size - x) / (10 * size), axis=-1), (size, 50))
                v = begin * np.exp((np.expand_dims(exp_v, axis=-1) + exp_vnoise) * end)
                yield x, v, {'label': r'$x_{' + str(i + 1) + r'}$'}
        engine.draw.plot_error_curves(func_gen(), y_log=True,
                                      y_error_method='minmax',
                                      xlabel='Step', ylabel=r'$\mathcal{L}$')
        engine.draw.plt.show()
        engine.draw.plot_error_curves(func_gen(), y_log=True,
                                      y_error_method='minmax', plot_method='fill',
                                      xlabel='Step', ylabel=r'$\mathcal{L}$')
        engine.draw.plt.show()

    @engine.draw.setFigure(style='classic', font_size=16)
    def test_distribution(self):
        def func_gen():
            size = 100
            x = np.arange(start=0, stop=size)
            for i in range(1):
                begin = 1 + 99.0 * np.random.rand()
                end = 2 + 10 * np.random.rand()
                exp_v = np.square((x - size) / size) - 1.0
                exp_vnoise = np.random.normal(0.0, np.expand_dims((size - x) / (10 * size), axis=-1), (size, 50))
                v = begin * np.exp((np.expand_dims(exp_v, axis=-1) + exp_vnoise) * end)
                yield x, v, {'label': r'$x_{' + str(i + 1) + r'}$'}
        engine.draw.plot_distribution_curves(func_gen(), method='mean', level=5, outlier=0.05,
                                             xlabel='Step', ylabel=r'$\mathcal{L}$', y_log=True)
        engine.draw.plt.show()

    def test_draw_context(self):
        with engine.draw.setFigure(style='classic', font_size=16, font_name='arial'):
            t = np.linspace(-10, 10, 100)
            plt.plot(t, 1 / (1 + np.exp(-t)))
            plt.title('In the context, font: arial.')
            plt.show()


class TestTools:
    '''Test functions for tools sub-module.
    '''
    def __init__(self, seed=None):
        self.random_rng = np.random.default_rng(seed=seed)

    def test_recorder(self):
        # Test epoch metrics.
        records = self.random_rng.normal(loc=0.0, scale=1.0, size=100)
        val2 = 1.0
        recorder = engine.tools.EpochMetrics(zip(('val1', 'val2'), ([0.0], [val2])))
        for r in records:
            recorder['val1'] = r
        print('utils.tools: EpochMetrics["val1"]={0}, np.mean(records)={1}.'.format(recorder['val1'], np.mean(records)))
        print('utils.tools: EpochMetrics["val2"]={0}, np.max(records)={1}.'.format(recorder['val2'], val2))

    def test_ctxwrapper(self):
        num_iters = 100
        with engine.tools.ContextWrapper(tqdm.tqdm(total=num_iters)) as tq:
            for i in range(num_iters):
                tq.update(1)
                time.sleep(0.001)


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
        '-udw', '--test_utl_draw', type=str2bool, nargs='?', const=True, default=False, metavar='bool',
        help='''Test the utils.draw module.'''
    )
    parser.add_argument(
        '-uto', '--test_utl_tools', type=str2bool, nargs='?', const=True, default=False, metavar='bool',
        help='''Test the utils.tools module.'''
    )
    if return_args:
        return parser.parse_args()
    else:
        return


# Test functions
def test_utl_draw():
    print('Compatibility test: mdnc.utils.draw.')
    tester = TestDraw()
    tester.test_all()
    tester.test_draw_context()


def test_utl_tools():
    print('Compatibility test: mdnc.utils.tools.')
    tester = TestTools()
    tester.test_recorder()
    tester.test_ctxwrapper()


registered_tests = {
    'test_utl_draw': test_utl_draw,
    'test_utl_tools': test_utl_tools
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
