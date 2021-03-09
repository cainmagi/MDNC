#!python
# -*- coding: UTF-8 -*-
'''
################################################################
# Utilities - Tools
# @ Modern Deep Network Toolkits for pyTorch
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.5+
#   numpy 1.13+
# Useful tools for data analysis and recording. This module is
# designed for a pyTorch-style substitute of keras.callbacks.
################################################################
'''

import numpy as np

__all__ = ['EpochMetrics', 'ContextWrapper']


class EpochMetrics(dict):
    '''A dictionary for storing metrics.
    '''
    def __init__(self, *args, **kwargs):
        '''Initialization
        Used as dict APIs, but provides a special argument.
        Arguments:
            reducer: a function, used for reducing the metrics, the default
                     value is np.mean.
        '''
        reducer = kwargs.pop('reducer', np.mean)
        super().__init__(*args, **kwargs)
        self.reducer = reducer

    def __setitem__(self, k, v=None):
        if v is None:
            super().__setitem__(k, list())
            return
        if isinstance(k, (int, float)):
            raise TypeError('Only allow setting scalar values.')
        log_list = super().get(k, list())
        log_list.append(v)
        super().__setitem__(k, log_list)

    def __getitem__(self, k):
        log_list = super().__getitem__(k)
        return self.reducer(log_list)

    def setdefault(self, k, default=None):
        self.__setitem__(k, default)

    def get(self, k, default=None):
        if k not in self:
            return default
        else:
            return self.__getitem__(k)

    def pop(self, k, default=None):
        log_list = super().pop(k, default)
        return self.reducer(log_list)

    def popitem(self, k):
        k, log_list = super().popitem(k)
        return k, self.reducer(log_list)

    def items(self):
        for k, log_list in super().items():
            yield k, self.reducer(log_list)

    def values(self):
        for v in super().values():
            yield self.reducer(v)


class ContextWrapper:
    '''A simple wrapper for adding context support to some special classes.
    For example, there is an instance f, it defines f.close(), but does
    not support the context. In this case, we could use this wrapper to
    add context support:
    ```python
    f = create_f(...)
    with mdnc.utils.tools.ContextWrapper(f) as fc:
        do some thing ...
    # When leaving the context, the f.close() method would be called
    # automatically.
    ```
    '''
    def __init__(self, instance, exit_method=None):
        '''Initialization
        Arguments:
            instance: an instance requring the context support.
            exit_method: a function, if not provided, would call the
                         instance.close() method during the exiting
                         stage. If provided, would call
                            exit_method(instance) instead.
        '''
        self.instance = instance
        self.exit_method = exit_method

    def __enter__(self):
        return self.instance

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.exit_method is None:
            self.instance.close()
        else:
            self.exit_method(self.instance)
