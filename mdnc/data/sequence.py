#!python
# -*- coding: UTF-8 -*-
'''
################################################################
# Data - sequence
# @ Modern Deep Network Toolkits for pyTorch
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6+
#   numpy 1.13+
# Optional package:
#   pyTorch 1.0.0+
# This module provides the fundamental `sequence` engine for
# pyTorch.
################################################################
'''

import os
import uuid
import warnings
import threading
import queue

import numpy as np

try:
    import torch
    import torch.multiprocessing as multiprocessing
    USE_TORCH_OUT = True
except ImportError:
    USE_TORCH_OUT = False
    import multiprocessing

__all__ = ['MSequence', 'MPSequence', 'MTSequence']


class _MSequence:
    def __init__(self, worker, is_out_list=True, out_func=None):
        if out_func is None:
            self._out_func = None
            self.out_func = self._preserve
        elif is_out_list:
            self._out_func = out_func
            self.out_func = self._post_proc
        else:
            self._out_func = None
            self.out_func = out_func
        self.worker = worker

    @staticmethod
    def _preserve(new_data):
        return new_data

    def _post_proc(self, new_data):
        return tuple(self._out_func(d) for d in new_data)

    def __call__(self, qi, qo):
        '''
        Target process (thread). Would be used internally.
        '''
        worker = self.worker()
        while True:
            inds = qi.get()
            if inds is None:
                # qi.task_done()
                qo.put(None)
                break
            data = worker[inds]
            data = self.out_func(data)
            qo.put(data)
            # qi.task_done()


class CudaConverter:
    '''A middle process for converting Tensor to cuda Tensor.
    This process should be only one.
    '''
    def __init__(self, is_out_list, device):
        self.device = torch.device(device)
        if is_out_list:
            self.out_func = self._convert_list
        else:
            self.out_func = self._convert_single

    def _convert_single(self, new_data):
        return new_data.to(self.device)

    def _convert_list(self, new_data):
        return tuple(d.to(self.device) for d in new_data)

    def __call__(self, qi, qo, to_workers=0):
        while True:
            data = qi.get()
            if data is None:
                to_workers -= 1
                if to_workers <= 0:
                    qo.put(None)
                    break
            else:
                qo.put(self.out_func(data))


class MSequence:
    '''MDNT sequence
    This class is a scheduler based on multi-threading/processing. It supports different
    workers and allows users to read datasets asynchronously and shuffle dataset
    randomly.
    '''

    def __init__(self, worker, dset_size,
                 num_workers=4, num_converters=None,
                 batch_size=32, buffer=10, shuffle=True,
                 thread_type='proc', out_type='cuda',
                 seed=None):
        '''
        Create the parser and its h5py file handle.
        Arguments:
            worker: An instance, with __getitem__() method implemented. This
                    instance would be copied and used as indexer for different
                    processes or threads.
            dset_size: the number of samples in the dataset. If given an array,
                       the array would be used as indices, the dset_size would
                       be the length of the array.
            num_workers: the number of workers.
            num_converters: the number of converters, only used when cuda is
                            enabled. If set None, would be determined by
                            num_workers.
            batch_size: number of samples in each batch.
            shuffle: if on, shuffle the data set at the end of each epoch.
            thread_type: the backend of the multi-threading, could be 'proc' or
                         'thread'.
            out_type: the output type. Could be 'cuda', 'cpu' or 'null'. If set
                      'null', the results would not be converted to tensor.
            seed: the seed used for shuffling the data. If not set, would use
                  random shuffle without seed.
        '''
        # Initialize multi-thread
        self.use_proc = thread_type == 'proc'
        if thread_type not in ('proc', 'thread'):
            raise TypeError('data.sequence: The given "thread_type" is invalid, should be "proc" or "thread", but given "{0}"'.format(thread_type))
        self.pool_ctx = None  # The spawn context
        self.pool_workers = list()  # List of processes for workers
        self.pool_converters = list()  # List of processes for converters
        self.qi = None  # Input queue
        self.qmid = None  # Converter (intermediate) queue, only used when cuda is enabled.
        self.qo = None  # Output queue

        # Check arguments
        if num_workers < 1:
            raise ValueError('data.sequence: The number of workers should be at least 1.')
        n_req = num_workers * batch_size
        self.worker = worker
        self.batch_size = batch_size
        self.n_w = num_workers
        if num_converters is None:
            num_converters = max(num_workers // 4, min(num_workers, 4))
        self.n_c = num_converters
        if self.n_w <= 0 or self.n_c <= 0:
            raise ValueError('data.sequence: The number of workers and converters should be at least 1.')
        if self.n_c > self.n_w:
            raise ValueError('data.sequence: The number of converters should not be more than the number of workers.')
        self.shuffle = shuffle
        if buffer < 1:
            raise ValueError('data.sequence: The buffer size should be at least 1, but given:', buffer)
        self.buffer = buffer

        # Create scheduling params.
        if isinstance(dset_size, (list, tuple, np.ndarray)):
            self._indices = dset_size
            self.dset_size = len(dset_size)
        else:
            self.dset_size = dset_size
            self._indices = self.__create_indices()  # Create indices
        if self.dset_size < n_req:
            raise ValueError('data.sequence: The dataset should contains at least {0} samples.'.format(n_req))
        self.length = int(np.ceil(self.dset_size / self.batch_size).astype(np.int))

        # Set Torch related APIs
        self.use_cuda = False
        if USE_TORCH_OUT:
            if isinstance(out_type, str):
                if out_type.startswith('cuda'):
                    proc_out_type = torch.FloatTensor
                    self.use_cuda = torch.cuda.is_available()
                elif out_type == 'cpu':
                    proc_out_type = torch.FloatTensor
                elif out_type == 'null':
                    proc_out_type = None
            elif out_type is None:
                proc_out_type = torch.FloatTensor
            else:
                proc_out_type = out_type
        else:
            if isinstance(out_type, (type(None), str)):
                proc_out_type = None
            else:
                proc_out_type = out_type
        is_out_list = self._find_out_type_type()
        self.manager = _MSequence(worker=self.worker, is_out_list=is_out_list, out_func=proc_out_type)
        self.converter = CudaConverter(is_out_list=is_out_list, device=out_type) if self.use_cuda else None
        self.__compat = None

        self.__in_context = False
        self.start_id = None
        self.start_current_mode = None
        self.start_test_args = None
        try:
            rng_func = np.random.default_rng
        except AttributeError:
            rng_func = np.random.RandomState
        if seed is not None:
            self.random_rng = rng_func(seed)
        else:
            self.random_rng = rng_func()

    def __create_indices(self):
        '''
        Create a list of indices, only need to be run for once.
        '''
        return np.arange(self.dset_size, dtype=np.int)

    def __shuffle(self):
        '''
        Resort the indices randomly.
        '''
        self.random_rng.shuffle(self._indices)

    def __len__(self):
        '''
        Automatically calculate the steps for iterate the whole dataset.
        '''
        return self.length

    def __get_indicies(self, bind):
        '''
        Get indices from the batch index.
        '''
        inds = self._indices[bind * self.batch_size:(bind + 1) * self.batch_size]
        return inds

    def _find_out_type_type(self):
        worker = self.worker()
        idxs = self.__get_indicies(0)
        data = worker[idxs]
        del worker
        if isinstance(data, (list, tuple)):
            return True
        else:
            return False

    def __check_state(self, cur_id):
        return cur_id == self.start_id

    def __generator(self):
        if self.pool_ctx is None:
            self.__start()

        cur_id = self.start_id
        s = 0
        if self.shuffle:
            self.__shuffle()
        for _ in range(min(max(1, self.buffer // 2), self.length)):
            idxs = self.__get_indicies(s)
            s += 1
            self.qi.put(idxs)

        t = 0
        while s < self.length:
            idxs = self.__get_indicies(s)
            s += 1
            data = self.qo.get()
            yield data
            if cur_id != self.start_id:
                raise InterruptedError('data.sequence: Current iteration has been interrupted by finish().')
            # self.qo.task_done()
            t += 1
            self.qi.put(idxs)

        while t < self.length:
            data = self.qo.get()
            yield data
            if cur_id != self.start_id:
                raise InterruptedError('data.sequence: Current iteration has been interrupted by finish().')
            # qo.task_done()
            t += 1

    def __generator_test(self):
        if self.shuffle:
            self.__shuffle()

        cur_id = self.start_id
        worker = self.manager.worker()
        for s in range(self.length):
            idxs = self.__get_indicies(s)
            data = worker[idxs]
            if self.start_test_args['use_out_type']:
                data = self.manager.out_func(data)
                if self.start_test_args['use_cuda']:
                    data = self.converter.out_func(data)
            yield data
            if cur_id != self.start_id:
                raise InterruptedError('data.sequence: Current iteration has been interrupted by finish().')

    @staticmethod
    def __check_compat():
        if os.name == 'nt':  # Windows
            return True
        elif os.name == 'posix':  # Linux
            return False
        else:  # We do not know
            return False

    def __enter__(self):
        if self.start_id is None:
            raise OSError('data.sequence: Should not enter a manager context before starting it. Try to use "with manager.start()" or "with manager.start_test()" to enter the context.')
        self.__in_context = True
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.__finish()
        self.__in_context = False

    def start(self, compat=None):
        '''Start the process pool.
        Could be used like:
        ```python
            manager.start()
            for ... in manager:
                ...
            manager.finish()
        ```
        Or using the context:
        ```python
            with manager.start() as m:
                for ... in m:
                    ...
        ```
        Running start() or start_test() would interrupt the started sequence.
        The cuda.Tensor could not be put into the queue on Windows (but on Linux we could), see
            https://pytorch.org/docs/stable/notes/windows.html#cuda-ipc-operations
        To solve this problem, we need to fall back to multi-threading for converter on Windows.
        Argument:
            compat: whether to fall back to multi-threading for the converter. If set None, the
                decision would be made by checking os.name.
        '''
        if self.__in_context:
            raise OSError('data.sequence: Should not start() the manager explicitly when it is already managed by a context. Try to exit the context before doing that.')
        self.__start(compat=compat)
        return self

    def start_test(self, test_mode='default'):
        '''Start the test mode.
        In the test mode, the process pool would not be open. All operations
        would be finished in the main thread. However, the random indices are
        still generated in the same seed of the parallel start() mode.
        Could be used like:
        ```python
            manager.start_test()
            for ... in manager:
                ...
            manager.finish()
        ```
        Or using the context:
        ```python
            with manager.start_test() as m:
                for ... in m:
                    ...
        ```
        Running start() or start_test() would interrupt the started sequence.
        Arguments:
            test_mode: Could be 'default', 'cpu', 'numpy'
                'default': the output would be converted as start() mode.
                'cpu': even set 'cuda' as output type, the testing output
                       would be still not converted to GPU.
                'numpy': would ignore all out_type configurations and
                         return the original output.
        '''
        if self.__in_context:
            raise OSError('data.sequence: Should not start() the manager explicitly when it is already managed by a context. Try to exit the context before doing that.')
        self.__start_test(test_mode=test_mode)
        return self

    def finish(self):
        '''Finish the process poll
        Would terminate all sub-processes and close the multiprocessing context.
        '''
        if self.__in_context:
            raise OSError('data.sequence: Should not finish() the manager explicitly when it is already managed by a context. Try to exit the context before doing that.')
        self.__finish()

    def __start(self, compat=None):
        # Check the unfinished processes
        if self.pool_ctx:
            self.finish()
        # Start a new context.
        self.pool_ctx = multiprocessing.get_context('spawn')
        self.start_id = uuid.uuid4()
        self.start_current_mode = 'normal'
        # Check compatibility
        self.__compat = self.__check_compat() if compat is None else bool(compat)
        if self.use_proc:
            if self.__compat:
                compat_Process = threading.Thread
                compat_Queue = queue.Queue
            else:
                compat_Process = self.pool_ctx.Process
                compat_Queue = self.pool_ctx.Queue
            Process = self.pool_ctx.Process
            Queue = self.pool_ctx.Queue
        else:
            Process, compat_Process = threading.Thread, threading.Thread
            Queue, compat_Queue = queue.Queue, queue.Queue
        # Begin to create Queues and Processes
        self.qi = Queue(maxsize=self.buffer)
        if self.use_cuda:
            self.qmid = Queue(maxsize=self.buffer)
            self.qo = compat_Queue(maxsize=self.buffer)
        else:
            self.qo = Queue(maxsize=self.buffer)
        qo_pool = self.qmid if self.use_cuda else self.qo
        for _ in range(self.n_w):
            th = Process(target=self.manager, args=(self.qi, qo_pool), daemon=True)
            th.start()
            self.pool_workers.append(th)
        # Create the Processes for the Converters
        if self.use_cuda:
            each_to_workers = self.n_w // self.n_c
            n_remains = self.n_w - each_to_workers * self.n_c
            nums_to_workers = (each_to_workers + 1, ) * n_remains + (each_to_workers, ) * (self.n_c - n_remains)
            for n2w in nums_to_workers:
                th = compat_Process(target=self.converter, args=(self.qmid, self.qo, n2w), daemon=True)
                th.start()
                self.pool_converters.append(th)

    def __start_test(self, test_mode='default'):
        self.start_id = uuid.uuid4()
        self.start_current_mode = 'test'
        self.start_test_args = {
            'use_cuda': (self.use_cuda and test_mode == 'default'),
            'use_out_type': (test_mode in ('cpu', 'default'))
        }

    def __finish(self):
        if self.pool_ctx:
            # Put the end signal to workers
            for _ in range(self.n_w):
                self.qi.put(None)
            # The workers would send their end singals to converters, wait them
            to_finished = self.n_c
            while to_finished > 0:
                v = self.qo.get()
                if v is None:
                    to_finished -= 1
            # Have finished all workers and converters, begin to clear queues.
            if self.use_proc:
                self.qi.close()
                if self.use_cuda:
                    self.qmid.close()
                    if not self.__compat:
                        self.qo.close()
                else:
                    self.qo.close()
            # Clear and terminate processes
            for _ in range(self.n_w):
                p = self.pool_workers.pop(0)
                if self.use_proc:
                    p.terminate()
                del p
            if self.use_cuda:
                for _ in range(self.n_c):
                    p = self.pool_converters.pop(0)
                    if self.use_proc and (not self.__compat):
                        p.terminate()
                    del p
            # Finally, clean all the memeory.
            self.pool_workers.clear()
            self.pool_converters.clear()
            del self.qi
            self.qi = None
            if self.use_cuda:
                del self.qmid
                self.qmid = None
            del self.qo
            self.qo = None
            del self.pool_ctx
            self.pool_ctx = None
        self.__compat = None
        self.start_id = None
        self.start_current_mode = None

    def __del__(self):
        self.__finish()  # Inject the finish into the garbage collection.

    def __iter__(self):
        if self.start_current_mode == 'test':
            return self.__generator_test()
        elif self.start_current_mode == 'normal':
            return self.__generator()
        else:
            warnings.warn(message='Has not started the sequence before. The start() is called implicitly.', category=UserWarning)
            return self.__generator()


class MPSequence(MSequence):
    '''MDNT processing sequence
    This class is a scheduler based on multi-threading. It supports different workers
    and allows users to read datasets asynchronously and shuffle dataset randomly.
    '''

    def __init__(self, worker, dset_size,
                 num_workers=4, num_converters=None,
                 batch_size=32, buffer=10, shuffle=True,
                 out_type='cuda', seed=None):
        '''
        Create the parser and its h5py file handle.
        Arguments:
            worker: a class type, or something like this for creating a reader.
                    The class should provide __getitem__ method.
            dset_size: the number of samples in the dataset.
            num_workers: the number of workers.
            num_converters: the number of converters, only used when cuda is
                            enabled. If set None, would be determined by
                            num_workers.
            batch_size: number of samples in each batch.
            shuffle: if on, shuffle the data set at the end of each epoch.
            out_type: the output type. Could be 'cuda', 'cpu' or 'null'. If set
                'null', the results would not be converted to tensor.
        '''
        super(MPSequence, self).__init__(worker=worker, dset_size=dset_size,
                                         num_workers=num_workers, num_converters=num_converters,
                                         batch_size=batch_size, buffer=buffer, shuffle=shuffle,
                                         thread_type='proc', out_type=out_type, seed=seed)


class MTSequence(MSequence):
    '''MDNT threading sequence
    This class is a scheduler based on multi-threading. It supports different workers
    and allows users to read datasets asynchronously and shuffle dataset randomly.
    '''

    def __init__(self, worker, dset_size,
                 num_workers=4, num_converters=None,
                 batch_size=32, buffer=10, shuffle=True,
                 out_type='cuda', seed=None):
        '''
        Create the parser and its h5py file handle.
        Arguments:
            worker: a class type, or something like this for creating a reader.
                    The class should provide __getitem__ method.
            dset_size: the number of samples in the dataset.
            num_workers: the number of workers.
            num_converters: the number of converters, only used when cuda is
                            enabled. If set None, would be determined by
                            num_workers.
            batch_size: number of samples in each batch.
            shuffle: if on, shuffle the data set at the end of each epoch.
            out_type: the output type. Could be 'cuda', 'cpu' or 'null'. If set
                'null', the results would not be converted to tensor.
        '''
        super(MTSequence, self).__init__(worker=worker, dset_size=dset_size,
                                         num_workers=num_workers, num_converters=num_converters,
                                         batch_size=batch_size, buffer=buffer, shuffle=shuffle,
                                         thread_type='thread', out_type=out_type, seed=seed)


if __name__ == '__main__':
    __spec__ = None  # Handle the error caused by pdb module.
