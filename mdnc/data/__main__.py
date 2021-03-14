#!python
# -*- coding: UTF-8 -*-
'''
################################################################
# Data - Compatibility tests
# @ Modern Deep Network Toolkits for pyTorch
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.5+
#   urllib3 1.26.2+
#   numpy 1.13+
#   scipy 1.0.0+
#   h5py 3.1.0+
#   tqdm 4.50.2+
#   matplotlib 3.1.1+
# Run the following command to perform the test:
# ```bash
# python -m mdnc.data
# ```
################################################################
'''

import os
import time
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt
import tqdm

from mdnc import __version__
import mdnc.data as engine


class TestSequenceWorker:
    '''An testing worker used by TestSequence
    '''
    def __getitem__(self, indx):
        # print('data.sequence: thd =', indx)
        return indx


class TestSequence:
    '''Test functions for sequence sub-module.
    '''
    def test_sequence(self):
        mts = engine.sequence.MPSequence(TestSequenceWorker, dset_size=512, batch_size=1, out_type='cuda', shuffle=False, num_workers=1)
        with mts.start() as m:
            print('data.sequence: Normal mode.')
            for i in m:
                print(i)
            print('data.sequence: sleep for 5s')
            time.sleep(5)
            print('data.sequence: out of sleep')
            for i in m:
                print(i)
        with mts.start_test() as m:
            print('data.sequence: Test mode.')
            for i in m:
                print(i)
            print('data.sequence: sleep for 5s')
            time.sleep(5)
            print('data.sequence: out of sleep')
            for i in m:
                print(i)
        # Test the unsafe operation.
        mts.start_test()
        try:
            for i in mts:
                print(i)
                if i > 256:
                    mts.finish()
        except InterruptedError as e:
            print('data.sequence: Skip InterruptedError:', e)
        finally:
            mts.finish()


class TestH5PY:
    '''Test functions for h5py sub-module.
    '''
    def __init__(self, temp_root='alpha'):
        self.root = temp_root
        os.makedirs(self.root, exist_ok=True)

    @staticmethod
    def preprocfunc(x, y):
        # print('data.h5py:', x.shape)
        return x, y

    def test_gparser(self):
        parser = engine.h5py.H5GParser(os.path.join(self.root, r'b'), ['one', 'zero'],
                                       batch_size=3, num_workers=4, shuffle=True, preprocfunc=self.preprocfunc)
        with parser.start() as p:
            for i, data in enumerate(p):
                print('data.h5py: Epoch 1, Batch {0}'.format(i), data[0].shape, data[1].shape)

            for i, data in enumerate(p):
                print('data.h5py: Epoch 2, Batch {0}'.format(i), data[0].shape, data[1].shape)

    def test_saver(self):
        saver = engine.h5py.H5SupSaver(enable_read=False)
        saver.config(logver=1, shuffle=True, fletcher32=True, compression='gzip')
        with saver.open(os.path.join(self.root, 'a')) as s:
            s.dump('one', np.ones([25, 20]), chunks=(1, 20))
            s.dump('zero', np.zeros([25, 10]), chunks=(1, 10))
        with saver.open(os.path.join(self.root, 'b'), enable_read=True) as s:
            s.dump('one', np.ones([25, 20]), chunks=(1, 20))
            s.dump('zero', np.zeros([25, 10]), chunks=(1, 10))
        with saver.open(os.path.join(self.root, 'c')) as s:
            s.dump('test1', np.zeros([100, 20]))
            gb = s['group1']
            with gb['group2'] as g:
                g.dump('test2', np.zeros([100, 20]))
                # gb.close()
                g.dump('test2', np.ones([100, 20]))
                g.attrs = {'new': 1}
                g.set_link('test3', '/test1')
            print('data.h5py: Check open: s["group1"]={0}, s["group1/group2"]={1}'.format(gb.is_open, g.is_open))

    def test_hcparser(self, save=False):
        if save:
            saver = engine.h5py.H5SupSaver(enable_read=False)
            saver.config(logver=1, shuffle=True, fletcher32=True, compression='gzip')
            with saver.open(os.path.join(self.root, 'test')) as s:
                s.dump('key1', np.arange(1000), chunks=(10,))
                s.dump('key2', np.arange(1000) + 1, chunks=(10,))
                s.dump('key3', np.ones([1000, 10]), chunks=(10, 10))
            with saver.open(os.path.join(self.root, 'test2')) as s:
                s.dump('key1', np.arange(3), chunks=(3,))
                s.dump('key2', np.arange(3) + 1, chunks=(3,))
                s.dump('key3', np.ones([3, 10]), chunks=(3, 10))
        else:
            parser = engine.h5py.H5CParser(os.path.join(self.root, 'test2'), keywords_sequence=['key1', 'key3'], keywords_single=['key2'],
                                           batch_size=1, sequence_size=5, sequence_position=0, sequence_padding='same',
                                           shuffle=False, preprocfunc=None, num_workers=1, num_buffer=1)
            with parser.start() as p:
                for i, data in enumerate(p):
                    d1, d2, d3 = data
                    print('data.h5py:', i, d1[:, :], d2.shape, d3)

    def test_splitter(self, mode='c'):
        if mode == 's':  # save
            saver = engine.h5py.H5SupSaver(enable_read=False)
            saver.config(logver=1, shuffle=True, fletcher32=True, compression='gzip')
            with saver.open(os.path.join(self.root, 'test')) as s:
                s.dump('key1', np.arange(1000), chunks=(10,))
                s.dump('key2', np.arange(1000) + 1, chunks=(10,))
                s.dump('key3', np.ones([1000, 10]))
        elif mode == 'c':  # convert
            converter = engine.h5py.H5SeqConverter()
            converter.config(logver=1, shuffle=True, fletcher32=True, compression='gzip')
            with converter.open(os.path.join(self.root, 'test'), os.path.join(self.root, 'test_seq')) as c:
                c.convert('key1')
                c.convert('key3')
                c.copy('key2')
        elif mode == 'g':  # test GParser
            gparser = engine.h5py.H5GParser(os.path.join(self.root, 'test_seq'), keywords=['key1', 'key2'],
                                            batch_size=4, shuffle=False, preprocfunc=None)
            try:
                with gparser.start(compat=None) as gp:
                    # gp.start()
                    for i, data in enumerate(gp):
                        print('data.h5py:', i, data)
                        if i > 10:
                            raise EOFError('Force the test to reach its end.')
            except EOFError as e:
                print('data.h5py: Skip EOFError:', e)
        elif mode == 'q':
            parser = engine.h5py.H5CParser(os.path.join(self.root, 'test_seq'), keywords_sequence=['key1', 'key3'], keywords_single=['key2'],
                                           batch_size=1, sequence_size=5, sequence_position=-1, sequence_padding='same',
                                           shuffle=False, preprocfunc=None, num_workers=10, num_buffer=1)
            with parser.start(compat=None) as p:
                N = 5
                for n in range(N + 1):
                    if n == 1:
                        cur_time = time.time()
                    for i, data in tqdm.tqdm(enumerate(p), total=len(p)):
                        d1, d2, d3 = data
                        time.sleep(0.01)
                        # print('data.h5py:', i, d1[:,:], d2.shape, d3)
                past = (time.time() - cur_time) / N
                print('data.h5py: Time consumption (parallel):', past)
            with parser.start_test() as p:
                N = 5
                for n in range(N + 1):
                    if n == 1:
                        cur_time = time.time()
                    for i, data in tqdm.tqdm(enumerate(p), total=len(p)):
                        d1, d2, d3 = data
                        time.sleep(0.01)
                        # print('data.h5py:', i, d1[:,:], d2.shape, d3)
                past = (time.time() - cur_time) / N
                print('data.h5py: Time consumption (single thread):', past)


class TestWebTools:
    '''Test functions for webtools sub-module.
    '''
    def __init__(self, temp_root='alpha'):
        self.root = temp_root
        self.token = engine.webtools.get_token(token='', silent=True)
        os.makedirs(self.root, exist_ok=True)

    def test_data_checker(self):
        set_list_file = os.path.join(self.root, 'web-data')
        engine.webtools.DataChecker.init_set_list(set_list_file)
        dc = engine.webtools.DataChecker(root=self.root, set_list_file=set_list_file, token=self.token)
        dc.add_query_file('dataset_file_name_01.txt')
        dc.query()

    def test_download_by_address(self):
        engine.webtools.download_tarball_link('https://github.com/cainmagi/Dockerfiles/releases/download/xubuntu-v1.5-u20.04/share-pixmaps.tar.xz', path=self.root, verbose=True)

    def test_download_by_info(self):
        engine.webtools.download_tarball(user='cainmagi', repo='Dockerfiles', tag='xubuntu-v1.5-u20.04', asset='xconfigs-u20-04.tar.xz', path=self.root, token=self.token, verbose=True)


class _ProcTest(engine.preprocs.ProcAbstract):
    def __init__(self, number=1, inds=None, parent=None):
        super().__init__(inds=inds, parent=parent)
        self.number = number

    def preprocess(self, x):
        print('data.preprocs: Pre,', self.number)
        return x

    def postprocess(self, x):
        print('data.preprocs: Post,', self.__p())
        return x

    def __p(self):
        return self.number


class _ProcTest2(_ProcTest):
    def postprocess(self, x):
        print('data.preprocs: Post,', self.__p())
        return x

    def __p(self):
        return self.number * 100


class TestPreProcs:
    '''Test functions for preprocs sub-module.
    '''
    def __init__(self, temp_root='alpha'):
        self.root = temp_root
        os.makedirs(self.root, exist_ok=True)
        self.file_name = os.path.join(self.root, 'test.pkl')

    def test_simple(self):
        x = _ProcTest2(number=3, parent=_ProcTest(number=2, parent=_ProcTest(number=1)))
        with open(self.file_name, 'wb') as f:
            pickle.dump(x, f)
        with open(self.file_name, 'rb') as f:
            y = pickle.load(f)
        y.postprocess(y.preprocess(1))

        x = engine.preprocs.ProcScaler(inds=1, parent=engine.preprocs.ProcScaler(inds=0))
        x.shift = 0
        with open(self.file_name, 'wb') as f:
            pickle.dump(x, f)
        with open(self.file_name, 'rb') as f:
            y = pickle.load(f)
        t = y.preprocess(np.arange(10), np.arange(10))
        t_b = y.postprocess(*t)
        print('data.preprocs: Stack manner,', t, t_b)

        x = engine.preprocs.ProcMerge(num_procs=2)
        x[:] = engine.preprocs.ProcScaler()
        x[1] = engine.preprocs.ProcScaler(shift=0)
        with open(self.file_name, 'wb') as f:
            pickle.dump(x, f)
        with open(self.file_name, 'rb') as f:
            y = pickle.load(f)
        t = y.preprocess(np.arange(10), np.arange(10))
        t_b = y.postprocess(*t)
        print('data.preprocs: Merge manner,', t, t_b)

    def test_1d(self):
        x = engine.preprocs.ProcPad(pad_width=((0, 0), (200, -200)), mode='reflect', parent=engine.preprocs.ProcNSTScaler(dim=1, kernel_length=257, parent=engine.preprocs.ProcNSTFilter1d(length=1024, filter_type='fft', band_low=3.0, band_high=15.0, nyquist=100)))
        with open(self.file_name, 'wb') as f:
            pickle.dump(x, f)
        with open(self.file_name, 'rb') as f:
            y = pickle.load(f)
        data = 1 + 0.01 * np.random.rand(1, 1024)
        t = y.preprocess(data)
        tb = y.postprocess(t)
        fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(12, 5))
        axs[0].plot(t[0])
        axs[1].plot(tb[0])
        axs[2].plot(data[0])
        axs[0].set_ylabel('Preprocessing')
        axs[1].set_ylabel('Inversed preprocessing')
        axs[2].set_ylabel('Raw data')
        plt.tight_layout()
        plt.show()

    def test_2d(self):
        x = engine.preprocs.ProcNSTScaler(dim=2, parent=engine.preprocs.ProcLifter(a=1.0, parent=engine.preprocs.ProcPad(pad_width=((0, 0), (10, -10), (-10, 10)), mode='constant', constant_values=0.0)))
        with open(self.file_name, 'wb') as f:
            pickle.dump(x, f)
        with open(self.file_name, 'rb') as f:
            y = pickle.load(f)
        data = np.random.rand(10, 30, 30)
        t = y.preprocess(data)
        t_b = y.postprocess(t)
        fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(12, 4))
        im1 = axs[0].imshow(t[2])
        im2 = axs[1].imshow(t_b[0])
        im3 = axs[2].imshow(data[0])
        fig.colorbar(im1, ax=axs[0], pad=0.1, orientation='horizontal')
        fig.colorbar(im2, ax=axs[1], pad=0.1, orientation='horizontal')
        fig.colorbar(im3, ax=axs[2], pad=0.1, orientation='horizontal')
        axs[0].set_ylabel('Preprocessing')
        axs[1].set_ylabel('Inversed preprocessing')
        axs[2].set_ylabel('Raw data')
        plt.tight_layout()
        plt.show()


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
        '-dseq', '--test_dat_sequence', type=str2bool, nargs='?', const=True, default=False, metavar='bool',
        help='''Test the data.sequence module.'''
    )
    parser.add_argument(
        '-dh5', '--test_dat_h5py', type=str2bool, nargs='?', const=True, default=False, metavar='bool',
        help='''Test the data.h5py module.'''
    )
    parser.add_argument(
        '-dwt', '--test_dat_webtools', type=str2bool, nargs='?', const=True, default=False, metavar='bool',
        help='''Test the data.webtools module.'''
    )
    parser.add_argument(
        '-dproc', '--test_dat_preprocs', type=str2bool, nargs='?', const=True, default=False, metavar='bool',
        help='''Test the data.preprocs module.'''
    )
    if return_args:
        return parser.parse_args()
    else:
        return


# Test functions
def test_dat_sequence():
    tester = TestSequence()
    tester.test_sequence()


def test_dat_h5py():
    tester = TestH5PY()
    # Test H5GParser
    tester.test_saver()
    tester.test_gparser()
    # Test H5CParser
    tester.test_hcparser(True)  # True, False
    tester.test_hcparser(False)
    # Test H5SeqConverter
    tester.test_splitter('s')  # s, c, g, q
    tester.test_splitter('c')
    tester.test_splitter('g')
    tester.test_splitter('q')


def test_dat_webtools():
    tester = TestWebTools()
    tester.test_download_by_address()
    tester.test_download_by_info()
    tester.test_data_checker()


def test_dat_preprocs():
    tester = TestPreProcs()
    tester.test_simple()
    tester.test_1d()
    tester.test_2d()


registered_tests = {
    'test_dat_sequence': test_dat_sequence,
    'test_dat_h5py': test_dat_h5py,
    'test_dat_webtools': test_dat_webtools,
    'test_dat_preprocs': test_dat_preprocs
}


if __name__ == '__main__':
    __spec__ = None  # Handle the error caused by pdb module.

    print('Compatibility test: mdnc.data. MDNC version: ', __version__)

    # Set parser and parse args.
    aparser = argparse.ArgumentParser(
        description='Compatibility test: mdnc.data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    args = vars(parse_args(aparser))

    if not any(args.values()):
        aparser.print_help()
    else:
        for k, req_run in args.items():
            if req_run:
                registered_tests[k]()
