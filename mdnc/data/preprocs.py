#!python
# -*- coding: UTF-8 -*-
'''
################################################################
# Data - preprocessors
# @ Modern Deep Network Toolkits for pyTorch
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.5+
#   numpy 1.13+
# This module provides some built-in pre-processors. These pre-
# processors support the cascading operation.
################################################################
'''

import collections
import numpy as np


class _ProcMemDict:
    '''A memory dict used for storing the intermediate results
    of pre-processors.
    '''
    def __init__(self):
        self.__mem = {0: dict()}
        self.__curdict = self.__mem[0]

    def clear(self):
        '''Clear the dictionary.'''
        self.__mem.clear()

    @property
    def curdict(self):
        '''Get current dictionary.'''
        return self.__curdict

    @curdict.setter
    def curdict(self, pos):
        '''Set the position of the current dictionary.'''
        if pos not in self.__mem:
            self.__mem[pos] = dict()
        self.__curdict = self.__mem[pos]

    def __setitem__(self, key, value):
        self.__curdict[key] = value

    def __getitem__(self, key):
        return self.__curdict[key]


class ProcAbstract:
    '''The basic processor class supporting cascading.
    Should be inherited like the following example:
    ```python
    class ProcExample(ProcAbstract):
        def __init__(self, ..., inds=inds, parent=None):
            super().__init__(inds=inds, parent=parent)
        def preprocess(self, ...):
            ...
        def postprocess(self, ...):
            ...
    ```
    The intertage has 3 requirements:
        1. The init method need to call the init method of the base
           class.
        2. The preprocess() and postprocess() methods need to be
           implemented.
    '''

    def __init__(self, inds=None, parent=None):
        '''Initialization
        Arguments:
            inds:   the indices of the positional arguments that would
                    be applied with the processing.
            parent: An instance of the ProcAbstract. This instance would
                    be used as the parent of the current instance.
        '''
        if parent is not None and not getattr(parent, '_ProcAbstract__isinitialized', False):
            raise TypeError('data.preprocs: The argument "parent" requires to be None or a sub-class of "ProcAbstract".')
        self.__parent = parent
        self.__inds = self.__init_inds(inds)
        self.__isinitialized = True
        self.__mem = _ProcMemDict()
        self.__stack_pre, self.__stack_post = self.__init_stack()

    @property
    def parent(self):
        return self.__parent

    def get_mem(self, key):
        return self.__mem[key]

    def set_mem(self, key, value):
        self.__mem[key] = value

    @staticmethod
    def __init_inds(inds):
        '''Arrange the indices.
        The integers would be used for indexing the list args.
        '''
        if inds is None:
            return None
        if not isinstance(inds, (list, tuple)):
            inds = (inds, )
        inds_args = list()
        for i in inds:
            if isinstance(i, int):
                inds_args.append(i)
        return inds_args

    def __init_stack(self):
        '''Initialize the stack of functions
        '''
        stack_pre = collections.deque()
        stack_post = collections.deque()
        # Get current stack
        stack_pre.append(self._ProcAbstract__preprocess_inds)
        stack_post.appendleft(self._ProcAbstract__postprocess_inds)
        # Get previous stack
        parent = self.parent
        while parent is not None:
            stack_pre.append(parent._ProcAbstract__preprocess_inds)
            stack_post.appendleft(parent._ProcAbstract__postprocess_inds)
            parent = parent.parent
        return stack_pre, stack_post

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.__init__(*args, **kwargs)
        isinit = getattr(instance, '_ProcAbstract__isinitialized', False)
        if not isinit:
            raise NotImplementedError('data.preprocs: This processor class has not been initialized by its super().__init__().')
        preprocess = super(ProcAbstract, instance).__getattribute__('preprocess')
        if preprocess is None or (not callable(preprocess)):
            raise NotImplementedError('data.preprocs: This processor has not implemented the required "preprocess()" method.')
        postprocess = super(ProcAbstract, instance).__getattribute__('postprocess')
        if postprocess is None or (not callable(postprocess)):
            raise NotImplementedError('data.preprocs: This processor has not implemented the required "postprocess()" method.')
        return super().__new__(cls)

    def __getattribute__(self, key):
        # print('data.preprocs: key',key)
        if key == 'preprocess':
            return self.__preproc_run
        elif key == 'postprocess':
            return self.__postproc_run
        else:
            return super(ProcAbstract, self).__getattribute__(key)

    def __preproc_run(self, *args):
        for st in self.__stack_pre:
            args = st(*args)
        if len(args) == 1:
            return args[0]
        else:
            return args

    def __postproc_run(self, *args):
        for st in self.__stack_post:
            args = st(*args)
        if len(args) == 1:
            return args[0]
        else:
            return args

    def __call__(self, *args):
        return self.preprocess(*args)

    def _ProcAbstract__preprocess_inds(self, *args):
        res_args = list()
        preprocess = super().__getattribute__('preprocess')
        if self.__inds is None:
            for i, x in enumerate(args):
                self.__mem.curdict = i
                res_args.append(preprocess(x))
        else:
            for i, x in enumerate(args):
                self.__mem.curdict = i
                if i in self.__inds:
                    res_args.append(preprocess(x))
                else:
                    res_args.append(x)
        return res_args

    def _ProcAbstract__postprocess_inds(self, *args):
        res_args = list()
        postprocess = super().__getattribute__('postprocess')
        if self.__inds is None:
            for i, x in enumerate(args):
                self.__mem.curdict = i
                res_args.append(postprocess(x))
        else:
            for i, x in enumerate(args):
                self.__mem.curdict = i
                if i in self.__inds:
                    res_args.append(postprocess(x))
                else:
                    res_args.append(x)
        return res_args


class ProcScaler(ProcAbstract):
    '''Scaler (Processor).
    Rescale the mean and std. values of the input data.
        y = (x - shift) / scale
    '''

    def __init__(self, shift=None, scale=None, axis=-1, inds=None, parent=None):
        '''Initialization.
        Arguments:
            shift:  the shifting parameter of the data. If set None,
                    would be calculated by the given axis (axes).
            scale:  the scaling parameter of the data. If set None,
                    would be calculated by the given axis (axes).
            axis:   the axis used for automatically calculating the
                    shift and scale value.
            inds:   the indices of the positional arguments that would
                    be applied with the processing.
            parent: An instance of the ProcAbstract. This instance would
                    be used as the parent of the current instance.
        '''
        super().__init__(inds=inds, parent=parent)
        self.shift = shift
        self.scale = scale
        self.axis = axis

    def preprocess(self, x):
        xmean = np.mean(x, axis=self.axis) if self.shift is None else self.shift
        xscale = np.amax(np.abs(x - xmean), axis=self.axis) if self.scale is None else self.scale
        self.set_mem('xmean', xmean)
        self.set_mem('xscale', xscale)
        return (x - xmean) / xscale

    def postprocess(self, x):
        xmean = self.get_mem('xmean')
        xscale = self.get_mem('xscale')
        x = x * xscale + xmean
        return x


class ProcNSTScaler(ProcAbstract):
    '''Non-stationary Scaler (Processor).
    Rescale the mean and std. values of the input data.
        y = (x - shift) / scale
    where shift, and scale are calculated by pooling.
    The codes for pooling is modified from the following material:
        https://stackoverflow.com/a/49317610
    '''

    def __init__(self, dim=2, kernel_length=9, epsilon=1e-6, inds=None, parent=None):
        '''Initialization.
        Arguments:
            shift:  the shifting parameter of the data. If set None,
                    would be calculated by the given axis (axes).
            scale:  the scaling parameter of the data. If set None,
                    would be calculated by the given axis (axes).
            axis:   the axis used for automatically calculating the
                    shift and scale value.
            inds:   the indices of the positional arguments that would
                    be applied with the processing.
            parent: An instance of the ProcAbstract. This instance would
                    be used as the parent of the current instance.
        '''
        super().__init__(inds=inds, parent=parent)
        self.__dim = dim
        self.__kernel_length = kernel_length
        self.epsilon = epsilon
        self.__kernel = None
        self.__pad = None
        self.__axes = None
        self.__set_kernel()

    @property
    def kernel_length(self):
        return self.__kernel_length

    @kernel_length.setter
    def kernel_length(self, value):
        self.__kernel_length = value
        self.__set_kernel()

    @property
    def dim(self):
        return self.__dim

    @dim.setter
    def dim(self, value):
        self.__dim = value
        self.__set_kernel()

    def __set_kernel(self):
        if isinstance(self.__kernel_length, (list, tuple)) and len(self.__kernel_length) == self.__dim:
            self.__kernel = tuple(*self.__kernel_length)
        else:
            self.__kernel = (self.__kernel_length, ) * self.__dim
        for k in self.__kernel:
            if k % 2 == 0:
                raise ValueError('data.preprocs: The kernel length need to be an odd number or a sequence with all elements odd.')
        self.__pad = ((0, 0),) + tuple(map(lambda k: (k // 2, k // 2), self.__kernel))  # First dimension is batch
        self.__axes = tuple(range(-self.__dim, 0, 1))

    def as_strided(self, arr):
        '''Get a strided sub-matrices view of an ndarray.
        See also skimage.util.shape.view_as_windows()
        '''
        dim = self.__dim
        s = arr.strides[-dim:]
        vshape = tuple(map(lambda m, n: (1 + m - n), arr.shape[-dim:], self.__kernel)) + tuple(self.__kernel)
        view_shape = arr.shape[:-dim] + vshape
        strides = arr.strides[:-dim] + (*s, *s)
        subs = np.lib.stride_tricks.as_strided(arr, view_shape, strides=strides)
        return subs

    def pooling(self, mat, method='max'):
        '''Make max pooling or average pooling with stride=1.
        Arguments:
            mat:    the input mini-batch.
            method: could be 'max' or 'mean'.
        '''
        mat_pad = np.pad(mat, self.__pad, mode='symmetric')
        view = self.as_strided(mat_pad)

        if method == 'max':
            result = np.nanmax(view, axis=self.__axes)
        else:
            result = np.nanmean(view, axis=self.__axes)
        return result

    def preprocess(self, x):
        xmean = self.pooling(x, method='mean')
        xscale = np.maximum(self.pooling(np.abs(x - xmean), method='max'), self.epsilon)
        self.set_mem('xmean', xmean)
        self.set_mem('xscale', xscale)
        return (x - xmean) / xscale

    def postprocess(self, x):
        xmean = self.get_mem('xmean')
        xscale = self.get_mem('xscale')
        x = x * xscale + xmean
        return x
