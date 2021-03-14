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
#   scipy 1.0.0+
# This module provides some built-in pre-processors. These pre-
# processors support the cascading operation.
################################################################
'''

import functools
import collections
import numpy as np

from scipy import signal

__all__ = ['ProcAbstract', 'ProcMerge',
           'ProcScaler', 'ProcNSTScaler', 'ProcFilter1d', 'ProcNSTFilter1d', 'ProcPad', 'ProcLifter']


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


class ProcAbstractMeta(type):
    '''Meta class of ProcAbstract.
    Should not be used by users.
    '''
    def __call__(cls, *args, **kwargs):
        '''Called when the ProcAbstract is initialized.'''
        obj = type.__call__(cls, *args, **kwargs)
        obj.init_verify()
        return obj


class ProcAbstract(metaclass=ProcAbstractMeta):
    '''The basic processor class supporting cascading and
    variable-level broadcasting..
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
    The intertage has 2 requirements:
        1. The init method need to call the init method of the base
           class.
        2. The preprocess() and postprocess() methods need to be
           implemented.
    '''

    def __init__(self, inds=None, parent=None, _disable_inds=False):
        '''Initialization
        Arguments:
            inds: the indices of the positional arguments that would
                  be applied with the processing.
            parent: An instance of the ProcAbstract. This instance would
                    be used as the parent of the current instance.
            _disable_inds: a flag for disabling the argument "inds". If
                           set, the arguments would not be dispatched
                           but passed to preprocess() and postprocess()
                           methods directly.
                           Should not be exposed to users, only use this
                           option for designing new processors.
        '''
        if parent is not None and not getattr(parent, '_ProcAbstract__isinitialized', False):
            raise TypeError('data.preprocs: The argument "parent" requires to be None or a sub-class of "ProcAbstract".')
        self.__parent = parent
        self.__disable_inds = _disable_inds
        self.__inds = None if self.__disable_inds else self.__init_inds(inds)
        self.__isinitialized = True
        self.__mem = _ProcMemDict()
        self.__stack_pre, self.__stack_post = self.__init_stack()

    def init_verify(self):
        '''Initialization verification
        This method is injected after the __init__() gets invoked
        automatically. It is used for verifying whether the inherited
        __init__ method is implemented correctly.
        '''
        isinit = getattr(self, '_ProcAbstract__isinitialized', False)
        if not isinit:
            raise NotImplementedError('data.preprocs: This processor class has not been initialized by its super().__init__().')
        try:
            preprocess = super().__getattribute__('preprocess')
        except AttributeError:
            preprocess = None
        if preprocess is None or (not callable(preprocess)):
            raise NotImplementedError('data.preprocs: This processor has not implemented the required "preprocess()" method.')
        try:
            postprocess = super().__getattribute__('postprocess')
        except AttributeError:
            postprocess = None
        if postprocess is None or (not callable(postprocess)):
            raise NotImplementedError('data.preprocs: This processor has not implemented the required "postprocess()" method.')
        self.__preproc_run.__func__.__name__ = 'preprocess'
        self.__postproc_run.__func__.__name__ = 'postprocess'

    @property
    def parent(self):
        return self.__parent

    @property
    def has_ind(self):
        proc = self
        has_ind_ = False
        while proc is not None:
            if proc._ProcAbstract__inds is not None or self._ProcAbstract__disable_inds:
                has_ind_ = True
                break
            proc = proc.parent
        return has_ind_

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
        stack_pre.appendleft(self._ProcAbstract__preprocess_inds)
        stack_post.append(self._ProcAbstract__postprocess_inds)
        # Get previous stack
        parent = self.parent
        while parent is not None:
            stack_pre.appendleft(parent._ProcAbstract__preprocess_inds)
            stack_post.append(parent._ProcAbstract__postprocess_inds)
            parent = parent.parent
        return stack_pre, stack_post

    def __getattribute__(self, key):
        # print('data.preprocs: key',key)
        if key == 'preprocess':
            return super().__getattribute__('_ProcAbstract__preproc_run')
        elif key == 'postprocess':
            return super().__getattribute__('_ProcAbstract__postproc_run')
        else:
            return super().__getattribute__(key)

    def __preproc_run(self, *args):
        '''Run pre-processing
        The inherit pre-processing method would be invoked here automatically.
        The arguments are the incoming variables for the pre-processing.
        '''
        for st in self.__stack_pre:
            args = st(*args)
        if len(args) == 1:
            return args[0]
        else:
            return args

    def __postproc_run(self, *args):
        '''Run post-processing
        The inherit post-processing method would be invoked here automatically.
        The arguments are the pre-processed variables. This method is the invert
        operator of preprocess().
        '''
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
        if self.__disable_inds:
            return preprocess(*args)
        elif self.__inds is None:
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
        return tuple(res_args)

    def _ProcAbstract__postprocess_inds(self, *args):
        res_args = list()
        postprocess = super().__getattribute__('postprocess')
        if self.__disable_inds:
            return postprocess(*args)
        elif self.__inds is None:
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
        return tuple(res_args)


class ProcMerge(ProcAbstract):
    '''Merge manager (Processor).
    This processor is designed for merging different processors by
    a more efficient way. For example,
    ```python
    p = ProcMerge([Proc1(...), Proc2(...)])
    ```
    is equivalent to
    ```python
    p = Proc1(..., inds=0, parent=Proc2(..., inds=1))
    ```
    This class should not be used if any sub-processor does not
    return the results with the same number of the input variables.
    '''
    def __init__(self, procs=None, num_procs=None, parent=None):
        '''Initialization
        Arguments:
            procs: a sequence of processors. Could be used for initializing
                   this merge processor.
            num_procs: the number of input of this processor. If not given,
                       would infer the number from the length of procs.
                       Either procs or num_procs needs to be specified.
            parent: An instance of the ProcAbstract. This instance would
                    be used as the parent of the current instance.
        '''
        super().__init__(parent=parent, _disable_inds=True)
        self.__num_procs, self.__procs_set = self.__init_with_procs(procs=procs, num_procs=num_procs)

    @property
    def num_procs(self):
        return self.__num_procs

    def __init_with_procs(self, procs, num_procs):
        if procs is not None and len(procs) > 0:
            len_procs = len(procs)
        else:
            len_procs = 0
        if num_procs is not None:
            if num_procs < len_procs:
                raise ValueError('data.preprocs: The argument "num_procs" need to be >= the length of the argument "len_procs".')
        else:
            num_procs = len_procs
        if num_procs == 0:
            raise ValueError('data.preprocs: Invalid configuration. When the argument "procs" is not given, the argument "num_procs" need to be given and > 0.')
        procs_set = dict()
        if procs is not None:
            for i, proc in enumerate(procs):
                if proc is not None and proc not in procs_set:
                    procs_set[proc] = set((i, ))
                else:
                    procs_set[proc].add(i)
        return num_procs, procs_set

    def __setitem__(self, idx, value):
        # Expand the idx to tuple
        if isinstance(idx, tuple):
            if not all(map(lambda x: (isinstance(x, int) and x >= 0 and x < self.__num_procs), idx)):
                return ValueError('data.preprocs: When using mulitple indicies, the indices should be all integers in [0, {n}).'.format(n=self.__num_procs))
        elif isinstance(idx, slice):
            x_start = idx.start if idx.start is not None else 0
            if x_start < 0:
                raise ValueError('data.preprocs: The slice range only support [0, {n})'.format(n=self.__num_procs))
            x_stop = idx.stop if idx.stop is not None else self.__num_procs
            if x_stop > self.__num_procs or x_stop <= x_start:
                raise ValueError('data.preprocs: The slice range only support [0, {n}), not supporting blank range.'.format(n=self.__num_procs))
            x_step = idx.step if idx.step else 1
            if x_step < 0 or x_step > (x_stop - x_start):
                raise ValueError('data.preprocs: The slice step should ensure that the range is not blank.')
            idx = tuple(range(x_start, x_stop, x_step))
        elif idx is Ellipsis:
            idx = tuple(range(self.__num_procs))
        elif isinstance(idx, int):
            idx = (idx, )
        else:
            raise TypeError('data.preprocs: The type of the given indicies is not supported.')
        if not isinstance(value, ProcAbstract):
            raise TypeError('data.preprocs: The value used for setting the item of ProcMerge requires to be a processor.')
        # merge idx into the procs_set.
        if value not in self.__procs_set:
            self.__procs_set[value] = set()
        proc_idx = self.__procs_set[value]
        for i in idx:
            proc_idx.add(i)
        for v, proc_idx in self.__procs_set.items():
            if v is not value:
                for i in idx:
                    proc_idx.discard(i)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise TypeError('data.preprocs: The index extraction only supports the int index.')
        for v, proc_idx in self.__procs_set.items():
            if idx in proc_idx:
                return v
        return None

    def preprocess(self, *args):
        res_args = list(args)
        for v, proc_idx in self.__procs_set.items():
            arg_inlist = tuple(args[i] for i in proc_idx)
            res = v.preprocess(*arg_inlist)
            if not isinstance(res, (tuple, list)):
                res = (res, )
            for i, r in zip(proc_idx, res):
                res_args[i] = r
        return res_args

    def postprocess(self, *args):
        res_args = list(args)
        for v, proc_idx in self.__procs_set.items():
            arg_inlist = tuple(args[i] for i in proc_idx)
            res = v.postprocess(*arg_inlist)
            if not isinstance(res, (tuple, list)):
                res = (res, )
            for i, r in zip(proc_idx, res):
                res_args[i] = r
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
        xmean = np.mean(x, axis=self.axis, keepdims=True) if self.shift is None else self.shift
        xscale = np.amax(np.abs(x - xmean), axis=self.axis, keepdims=True) if self.scale is None else self.scale
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

    def __init__(self, dim, kernel_length=9, epsilon=1e-6, inds=None, parent=None):
        '''Initialization.
        Arguments:
            dim: the dimension of the input data (to be normalized).
            kernel_length: the length of the non-stationary sldiing
                           window.
            epsilon: the lower bound of the divisor used for scaling.
            inds:   the indices of the positional arguments that would
                    be applied with the processing.
            parent: An instance of the ProcAbstract. This instance would
                    be used as the parent of the current instance.
        '''
        super().__init__(inds=inds, parent=parent)
        if dim not in (1, 2, 3):
            raise ValueError('data.preprocs: The argument "dim" requires to be 1, 2, or 3.')
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


class _ProcFilter1d:
    '''Use stationary band-pass filter to process data.
    This is the implementation of the 1D IIR band-pass filters (also supports
    low-pass or high-pass filter).
    This class is designed for building the ProcFilter1d and ProcNSTFilter1d,
    should not be used by users.
    '''
    def __init__(self, axis=-1, band_low=3.0, band_high=15.0, nyquist=500.0,
                 filter_type='butter', out_type='sosfilt2', filter_args=None):
        self.axis = int(axis)
        self.band_low = float(band_low) if band_low is not None else None
        self.band_high = float(band_high) if band_high is not None else None
        self.nyquist = float(nyquist)
        self.filter_type = filter_type
        self.out_type = out_type
        self.filter_args = self.__use_recommend_filter_args()
        if filter_args is not None:
            if isinstance(filter_args, dict):
                raise TypeError('data.preprocs: The argument "filter_args" should be a dict or None.')
            self.filter_args.update(filter_args)

        if self.filter_type == 'butter':
            self.create_butter()
        elif self.filter_type in ('cheby1', 'cheby2'):
            self.create_chebyshev()
        elif self.filter_type == 'ellip':
            self.create_elliptic()
        elif self.filter_type == 'bessel':
            self.create_bessel()
        else:
            raise ValueError('data.preprocs: The argument "filter_type" is not correct, should be selected from: "butter", "cheby1", "cheby2", "ellip", "bessel".')

    def __call__(self, data):
        data_res = self.filt(x=data)
        return data_res

    def __use_recommend_filter_args(self):
        default_args = {
            'order': 10,
            'ripple': 1,  # Maximum ripple (cheby1, ellip).
            'attenuation': 40
        }
        if self.filter_type in ('butter', 'bessel'):
            default_args.update({
                'order': 10
            })
        elif self.filter_type == 'cheby1':
            default_args.update({
                'order': 4,
                'ripple': 5
            })
        elif self.filter_type == 'cheby2':
            default_args.update({
                'order': 10,
                'attenuation': 40
            })
        elif self.filter_type == 'ellip':
            default_args.update({
                'order': 4,
                'ripple': 5,
                'attenuation': 40
            })
        return default_args

    def create_band(self):
        if self.band_low is not None and self.band_high is not None:
            wn = (self.band_low, self.band_high)
            mode = 'bandpass'
        elif self.band_low is not None:
            wn = self.band_low
            mode = 'highpass'
        elif self.band_high is not None:
            wn = self.band_high
            mode = 'lowpass'
        else:
            raise TypeError('data.preprocs: The cut-off frequencies for the band-pass filter are not specified.')
        return wn, mode

    def create_filter(self, filt):
        if self.out_type == 'sos':
            self.filt = functools.partial(signal.sosfilt, sos=filt, axis=self.axis)
        elif self.out_type == 'ba':
            self.filt = functools.partial(signal.lfilter, b=filt[0], a=filt[1], axis=self.axis)
        elif self.out_type == 'sosfilt2':
            self.filt = functools.partial(signal.sosfiltfilt, sos=filt, axis=self.axis)
        elif self.out_type == 'filt2':
            self.filt = functools.partial(signal.filtfilt, b=filt[0], a=filt[1], axis=self.axis)
        else:
            raise ValueError('data.preprocs: The out_type is not correct.')

    @staticmethod
    def map_out_type(out_type):
        if out_type in ('ba', 'filt2'):
            return 'ba'
        elif out_type in ('sos', 'sosfilt2'):
            return 'sos'
        else:
            raise TypeError('data.preprocs: The argument "out_type" should be "ba", "filt2" or "sos".')

    def create_butter(self):
        '''Butterworth filter'''
        wn, mode = self.create_band()
        filt = signal.butter(N=self.filter_args['order'], Wn=wn, btype=mode, fs=self.nyquist, output=self.map_out_type(self.out_type))
        self.create_filter(filt)

    def create_chebyshev(self):
        '''Chebyshev type I/II filter'''
        wn, mode = self.create_band()
        if self.filter_type == 'cheby1':
            filt = signal.cheby1(N=self.filter_args['order'], rp=self.filter_args['ripple'], Wn=wn, btype=mode, fs=self.nyquist, output=self.map_out_type(self.out_type))
        elif self.filter_type == 'cheby2':
            filt = signal.cheby2(N=self.filter_args['order'], rs=self.filter_args['attenuation'], Wn=wn, btype=mode, fs=self.nyquist, output=self.map_out_type(self.out_type))
        else:
            raise ValueError('data.preprocs: The argument "filter_type" should be "cheby1" or "cheby2".')
        self.create_filter(filt)

    def create_elliptic(self):
        '''Elliptic filter'''
        wn, mode = self.create_band()
        filt = signal.ellip(N=self.filter_args['order'], rp=self.filter_args['ripple'], rs=self.filter_args['attenuation'], Wn=wn, btype=mode, fs=self.nyquist, output=self.map_out_type(self.out_type))
        self.create_filter(filt)

    def create_bessel(self):
        '''Bessel/Thomson filter'''
        wn, mode = self.create_band()
        filt = signal.bessel(N=self.filter_args['order'], Wn=wn, btype=mode, norm='phase', fs=self.nyquist, output=self.map_out_type(self.out_type))
        self.create_filter(filt)


class _ProcNSTFilter1d:
    '''Use non-stationary band-pass filter and taping window to process data.
    This is the implementation of the 1D IIR band-pass filters (also supports
    low-pass or high-pass filter) on the non-stationary data.
    This class is designed for building the ProcNSTFilter1d, should not be
    used by users.
    '''
    def __init__(self, axis=-1, length=1024, patch_length=128, patch_step=64,
                 kaiser_coef=1.0, band_low=3.0, band_high=15.0, nyquist=500.0,
                 filter_type='butter', out_type='sosfilt2', filter_args=None):
        self.axis = int(axis)
        self.length = int(length)
        self.patch_length = int(patch_length)
        self.patch_step = int(patch_step)
        self.kaiser_coef = float(kaiser_coef)
        if self.length < self.patch_length or self.patch_length < self.patch_step or self.patch_step < 0:
            raise ValueError('data.preprocs: The arguments need to satisfy "length" > "patch_length" > "patch_step" > 0.')
        if self.kaiser_coef < 0:
            raise ValueError('data.preprocs: The argument "kaiser_coef" should >= 0.')
        self.band_low = float(band_low) if band_low is not None else None
        self.band_high = float(band_high) if band_high is not None else None
        self.nyquist = float(nyquist)
        if self.band_low is None and self.band_high is None:
            raise ValueError('data.preprocs: The cut-off frequencies for the band-pass filter are not specified.')
        self.create_patches()
        if filter_type not in (None, 'null', 'fft', 'none'):
            self.filt = _ProcFilter1d(axis=axis, band_low=band_low, band_high=band_high, nyquist=nyquist,
                                      filter_type=filter_type, out_type=out_type, filter_args=filter_args)
        else:
            self.filt = self.filter_fft

    def create_patches(self):
        patch, step = self.patch_length, self.patch_step
        length = self.length
        N = int(np.ceil((length - patch) / step))
        self.patches = list()
        for i in range(N - 1):
            self.patches.append((i * step, i * step + patch))
        self.patches.append((length - patch, length))
        self.win = np.kaiser(patch, self.kaiser_coef * np.pi)
        axis_freq = np.fft.rfftfreq(patch, d=1 / self.nyquist)
        max_freq = axis_freq[-1]
        n_freq = len(axis_freq)
        self.band_low_d = int(np.ceil(self.band_low / max_freq * n_freq)) if self.band_low is not None else None
        self.band_high_d = int(np.ceil(self.band_high / max_freq * n_freq)) if self.band_high is not None else None
        self.win_vec = np.zeros(length)
        for ind_l, ind_r in self.patches:
            self.win_vec[ind_l:ind_r] += self.win
        self.patch_len = patch

    def __call__(self, data):
        data_res = np.zeros_like(data)
        for ind_l, ind_r in self.patches:
            slices_i = [slice(None)] * data.ndim
            slices_i[self.axis] = slice(ind_l, ind_r)
            dwin = data[tuple(slices_i)]
            data_res[tuple(slices_i)] += self.filt(dwin)
        dshape = np.ones(data_res.ndim, dtype=np.int).tolist()
        dshape[self.axis] = len(self.win_vec)
        data_res = data_res / np.reshape(self.win_vec, dshape)
        return data_res

    def filter_fft(self, dwin):
        D = np.fft.rfft(dwin, axis=self.axis)
        if self.band_low_d is not None:
            slices_low = [slice(None)] * dwin.ndim
            slices_low[self.axis] = slice(None, self.band_low_d)
            D[tuple(slices_low)] = 0.0
        if self.band_high_d is not None:
            slices_high = [slice(None)] * dwin.ndim
            slices_high[self.axis] = slice(self.band_high_d, None)
            D[tuple(slices_high)] = 0.0
        dshape = np.ones(D.ndim, dtype=np.int).tolist()
        dshape[self.axis] = len(self.win)
        return np.fft.irfft(D, n=self.patch_len, axis=self.axis) * np.reshape(self.win, dshape)


class ProcFilter1d(ProcAbstract):
    '''Use stationary band-pass filter to process data.
    This is the implementation of the 1D IIR band-pass filters (also supports
    low-pass or high-pass filter).
    Plese pay attention to the results. This operation is not invertible, and
    the postprocess() would do nothing.
    '''
    def __init__(self, axis=-1, band_low=3.0, band_high=15.0, nyquist=500.0,
                 filter_type='butter', out_type='sosfilt2', filter_args=None,
                 inds=None, parent=None):
        '''Initialization
        Arguments:
            axis: the axis where we apply the 1D filter.
            band_low: the lower cut-off frequency. If only set this value,
                      the filter become high-pass.
            band_high: the higher cut-off frequency. If only set this value,
                       the filter become low-pass.
            nyquist: the nyquist frequency of the data.
            filter_type: the IIR filter type, could be
                         - butter - cheby1, - cheby2, - ellip, - bessel
            out_type: the output type, could be
                      - sosfilt2, - filt2, - sos, - ba
            filter_args: a dictionary including other filter arguments, not
                         all arguments are required for each filter. See the
                         scipy documents to view details. Support arguments:
                         - order, - ripple, - attenuation
            inds:   the indices of the positional arguments that would
                    be applied with the processing.
            parent: An instance of the ProcAbstract. This instance would
                    be used as the parent of the current instance.
        '''
        super().__init__(inds=inds, parent=parent)
        self.filt = _ProcFilter1d(axis=axis, band_low=band_low, band_high=band_high, nyquist=nyquist,
                                  filter_type=filter_type, out_type=out_type, filter_args=filter_args)

    def preprocess(self, x):
        return self.filt(x)

    def postprocess(self, x):
        return x


class ProcNSTFilter1d(ProcAbstract):
    '''Use non-stationary band-pass filter and taping window to process data.
    This is the implementation of the 1D IIR band-pass filters (also supports
    low-pass or high-pass filter) on the non-stationary data.
    Plese pay attention to the results. This operation is not invertible, and
    the postprocess() would do nothing.
    '''
    def __init__(self, axis=-1, length=1024, patch_length=128, patch_step=64,
                 kaiser_coef=1.0, band_low=3.0, band_high=15.0, nyquist=500.0,
                 filter_type='butter', out_type='sosfilt2', filter_args=None,
                 inds=None, parent=None):
        '''Initialization
        Arguments:
            axis: the axis where we apply the 1D filter.
            length: the length of the to be processed data.
            patch_length: the length of each 1D patch.
            patch_step: the step of the sliding window.
            kaiser_coef: the coefficent of the Kaiser window for each patch.
            band_low: the lower cut-off frequency. If only set this value,
                      the filter become high-pass.
            band_high: the higher cut-off frequency. If only set this value,
                       the filter become low-pass.
            nyquist: the nyquist frequency of the data.
            filter_type: the IIR filter type, could be
                         - butter - cheby1, - cheby2, - ellip, - bessel
                         the FIR filter type, could be
                         - fft
            out_type: the output type, could be
                      - sosfilt2, - filt2, - sos, - ba
            filter_args: a dictionary including other filter arguments, not
                         all arguments are required for each filter. See the
                         scipy documents to view details. Support arguments:
                         - order, - ripple, - attenuation
            inds:   the indices of the positional arguments that would
                    be applied with the processing.
            parent: An instance of the ProcAbstract. This instance would
                    be used as the parent of the current instance.
        '''
        super().__init__(inds=inds, parent=parent)
        self.filt = _ProcNSTFilter1d(axis=axis, length=length, patch_length=patch_length, patch_step=patch_step,
                                     kaiser_coef=kaiser_coef, band_low=band_low, band_high=band_high, nyquist=nyquist,
                                     filter_type=filter_type, out_type=out_type, filter_args=filter_args)

    def preprocess(self, x):
        return self.filt(x)

    def postprocess(self, x):
        return x


class ProcPad(ProcAbstract):
    '''Use np.pad to pad the data.
    Support all np.pad options. This processor also support cropping. If any element
    in the argument "pad_width" is negative, would perform cropping on that axis.
    For example:
    ```python
    p = ProcPad(pad_width=((5, -5),))
    y = p(x)  # x.shape=(20,), y.shape=(20,)
    ```
    In this case, the data is padded by 5 samples at the beginning, but cropped 5
    samples at the end.
    This operator is not invertible when cropping is applied. The postprocess would
    try to use padding for solving this case.
    '''
    def __init__(self, pad_width, inds=None, parent=None, **kwargs):
        '''Initialization
        Arguments:
            pad_width: number of values padded to the edges of each axis. Different
                       from the original np.pad API, this argument supports negative
                       values. A negative width represents a cropping size.
            inds:   the indices of the positional arguments that would
                    be applied with the processing.
            parent: An instance of the ProcAbstract. This instance would
                    be used as the parent of the current instance.
            **kwargs: other keywords could be referred here:
                https://numpy.org/doc/stable/reference/generated/numpy.pad.html
        '''
        super().__init__(inds=inds, parent=parent)
        self.__pad_width, self.__crop_width = self.__split_pad_width(pad_width)
        self.__pad_width_ = pad_width
        self.func_pad = functools.partial(np.pad, **kwargs)

    @staticmethod
    def __split_pad_width(pad_width):
        # Split the crop_width from pad_width
        if isinstance(pad_width, int):
            if pad_width < 0:
                return 0, abs(pad_width)
            else:
                return pad_width, 0
        elif isinstance(pad_width, (list, tuple)):
            if len(pad_width) == 1 and isinstance(pad_width[0], int):
                pad_width = pad_width[0]
                if pad_width < 0:
                    return 0, abs(pad_width)
                else:
                    return pad_width, 0
            elif len(pad_width) == 2 and all(map(lambda x: isinstance(x, int), pad_width)):
                crop_width = [0, 0]
                pad_width_ = [0, 0]
                for i, p in enumerate(pad_width):
                    if p > 0:
                        pad_width_[i] = p
                    else:
                        crop_width[i] = abs(p)
                return tuple(pad_width_), tuple(crop_width)
            elif all(map(lambda x: (isinstance(x, (list, tuple)) and (len(x) == 2)), pad_width)):
                crop_width = list()
                pad_width_ = list()
                for pw in pad_width:
                    cw = [0, 0]
                    pw_ = [0, 0]
                    if all(map(lambda x: isinstance(x, int), pw)):
                        for i, p in enumerate(pw):
                            if p > 0:
                                pw_[i] = p
                            else:
                                cw[i] = abs(p)
                        crop_width.append(tuple(cw))
                        pad_width_.append(tuple(pw_))
                return tuple(pad_width_), tuple(crop_width)
            else:
                raise ValueError('data.preprocs: the crop arguments could not get separated from the pad arguments. The given arguments "pad_width" may be not valid.')

    @property
    def pad_width(self):
        return object.__getattribute__(self, '_ProcPad__pad_width_')

    @pad_width.setter
    def pad_width(self, value):
        self.__pad_width, self.__crop_width = self.__split_pad_width(value)
        self.__pad_width_ = value

    @staticmethod
    def crop(x, crop_width):
        ndim = x.ndim
        if isinstance(crop_width, int):
            if crop_width > 0:
                crop_slices = (slice(crop_width, (-crop_width) if crop_width > 0 else None), ) * ndim
                return x[crop_slices]
        elif isinstance(crop_width, (tuple, list)):
            if len(crop_width) == 2 and all(map(lambda x: isinstance(x, int), crop_width)):
                crop_slices = (slice(crop_width[0], (-crop_width[1]) if crop_width[1] > 0 else None), ) * ndim
                return x[crop_slices]
            if len(crop_width) != ndim:
                raise ValueError('data.preprocs: the input data does not correspond to the shape of the crop configurations.')
            crop_slices = tuple(slice(cw[0], (-cw[1]) if cw[1] > 0 else None) for cw in crop_width)
            return x[crop_slices]
        else:
            return x

    def preprocess(self, x):
        x = self.func_pad(x, pad_width=self.__pad_width)
        x = self.crop(x, self.__crop_width)
        return x

    def postprocess(self, x):
        x = self.func_pad(x, pad_width=self.__crop_width)
        x = self.crop(x, self.__pad_width)
        return x


class ProcLifter(ProcAbstract):
    '''Use log lifting function to enhance the data.
        x = sign(x) * log(1 + a * abs(x))
    where a is the lifting coefficient given by users.
    '''
    def __init__(self, a, inds=None, parent=None):
        super().__init__(inds=inds, parent=parent)
        self.a = float(a)
        if self.a <= 0:
            raise ValueError('The argument "a" requires to be a postive value.')

    def preprocess(self, x):
        return np.sign(x) * np.log(1.0 + self.a * np.abs(x))

    def postprocess(self, x):
        return np.sign(x) * (np.exp(np.abs(x)) - 1.0) / self.a
