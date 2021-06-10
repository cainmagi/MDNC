#!python
# -*- coding: UTF-8 -*-
'''
################################################################
# Data - h5py
# @ Modern Deep Network Toolkits for pyTorch
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.5+
#   numpy 1.13+
#   h5py 3.1.0+
# Provide parallel utilities wrapped h5py. The data manager
# handle for MDNC include the implementations of the following
# protocols:
# Data format converter: H5Converter, H5SeqConverter
# Incremental dataset saver: H5SupSaver
# Data loader: H5GParser, H5RParser, H5CParser
################################################################
'''

import os
import io
import abc
import functools
import json

import h5py
import numpy as np

from . import sequence

__all__ = ['H52TXT', 'H52BIN', 'H5Converter', 'H5SeqConverter',
           'H5SupSaverGroup', 'H5SupSaver',
           'H5GParser', 'H5CParser', 'H5RParser']


class H52TXT:
    '''An example of converter between HDF5 and TXT'''
    def read(self, file_name):
        '''read function, for converting TXT to HDF5.
        file_name is the name of the single input file'''
        with open(os.path.splitext(file_name)[0] + '.txt', 'r') as f:
            sizeText = io.StringIO(f.readline())
            sze = np.loadtxt(sizeText, dtype=np.int)
            data = np.loadtxt(f, dtype=np.float32)
            return np.reshape(data, sze)

    def write(self, h5data, file_name):
        '''write function, for converting HDF5 to TXT.
        file_name is the name of the single output file.'''
        with open(os.path.splitext(file_name)[0] + '.txt', 'w') as f:
            np.savetxt(f, np.reshape(h5data.shape, (1, h5data.ndim)), fmt='%d')
            if h5data.ndim > 1:
                for i in range(h5data.shape[0]):
                    np.savetxt(f, h5data[i, ...].ravel(), delimiter='\n')
            else:
                np.savetxt(f, h5data[:].ravel(), delimiter='\n')


class H52BIN:
    '''An example of converter between HDF5 and bin file'''
    def read(self, file_name):
        '''read function, for converting bin file to HDF5.
        file_name is the name of the single input file'''
        with open(os.path.splitext(file_name)[0] + '.bin', 'rb') as f:
            ndims = np.fromfile(f, dtype=np.int, count=1)[0]
            sze = np.fromfile(f, dtype=np.int, count=ndims)
            data = np.fromfile(f, dtype=np.float32, count=-1)
            return np.reshape(data, sze)

    def write(self, h5data, file_name):
        '''write function, for converting HDF5 to bin file.
        file_name is the name of the single output file.'''
        with open(os.path.splitext(file_name)[0] + '.bin', 'wb') as f:
            get_ndim = np.array(h5data.ndim)
            get_shape = np.array(h5data.shape)
            get_ndim.tofile(f)
            get_shape.tofile(f)
            if get_ndim > 1:
                for i in range(get_shape[0]):
                    h5data[i, ...].ravel().astype(np.float32).tofile(f)
            else:
                h5data[:].ravel().astype(np.float32).tofile(f)


class H5Converter:
    '''Conversion between HDF5 data and other formats.
    The "other formats" would be arranged in to form of several
    folders and files. Each data group would be mapped into a
    folder, and each dataset would be mapped into a file.
    '''
    def __init__(self, file_name, oformat, to_other=True):
        '''Initialization and set format.
        Arguments:
            file_name: a path where we find the dataset. If the
                       conversion is h52other, the path should
                       refer a folder containing several subfiles,
                       otherwise, it should refer an HDF5 file.
            oformat:   the format function for a single dataset,
                       it could be provided by users, or use the
                       default configurations. (avaliable: 'txt',
                       'bin'.)
            to_other:  the flag for conversion mode. If set True,
                       the mode would be h52other, i.e. an HDF5
                       set would be converted into other formats.
                       If set False, the conversion would be
                       reversed.
        '''
        self.__read = (not to_other)
        self.folder = os.path.splitext(file_name)[0]
        if not self.__read:
            if not os.path.isfile(file_name):
                if (os.path.isfile(file_name + '.h5')):
                    file_name += '.h5'
                    self.folder = os.path.splitext(file_name)[0]
                else:
                    raise FileNotFoundError('data.h5py: Could not read the HDF5 dataset: {0}.'.format(file_name))
            if os.path.exists(self.folder):
                raise FileExistsError('data.h5py: Could not write to the folder: {0}, because it already exists.'.format(self.folder))
        else:
            if not os.path.isdir(file_name):
                raise FileNotFoundError('data.h5py: Could not open the folder {0}.'.format(file_name))
            if os.path.exists(file_name + '.h5'):
                raise FileExistsError('data.h5py: Could not write to the HDF5 dataset {0}.h5, because it already exists.'.format(file_name))
            self.folder = file_name
            file_name = file_name + '.h5'
        self.f = h5py.File(file_name, 'w' if self.__read else 'r')

        if oformat == 'txt':
            self.__func = H52TXT()
        elif oformat == 'bin':
            self.__func = H52BIN()
        else:
            if self.__read:
                if not hasattr(oformat, 'write'):
                    raise AttributeError('data.h5py: The "oformat" should contains the write method for applying the conversion.')
            else:
                if not hasattr(oformat, 'read'):
                    raise AttributeError('data.h5py: The "oformat" should contains the read method for applying the conversion.')
            self.__func = oformat

    @staticmethod
    def __h5iterate(g, func):
        if isinstance(g, h5py.Dataset):
            func(g)
        else:
            for item in g:
                H5Converter.__h5iterate(g[item], func)

    def __savefunc(self, g):
        path = os.path.join(self.folder, g.name.replace(':', '-')[1:])
        folder = os.path.split(path)[0]
        if not os.path.isdir(folder):
            os.makedirs(folder)
        self.__func.write(g, path)
        print('data.h5py: Have dumped {0}'.format(g.name))

    def __h52other(self):
        self.__h5iterate(self.f, self.__savefunc)

    def __other2h5(self):
        for root, _, files in os.walk(self.folder, topdown=False):
            for name in files:
                dsetName = '/' + os.path.relpath(os.path.join(root, os.path.splitext(name)[0]), start=self.folder).replace('\\', '/')
                self.f.create_dataset(dsetName, data=self.__func.read(os.path.join(root, name)))
                print('data.h5py: Have dumped {0}'.format(dsetName))

    def convert(self):
        '''Perform the data conversion.'''
        if self.__read:
            self.__other2h5()
        else:
            self.__h52other()


class H5SeqConverter:
    '''Convert any supervised .h5 data file into sequence version.
    This class allows users to choose some keywords and convert them
    into sequence version. Those keywords would be saved as in the
    format of continuous sequence. It could serve as a random
    splitter for preparing the training of LSTM.
    '''
    def __init__(self, file_in_name=None, file_out_name=None):
        '''
        Create the .h5 file while initialization.
        Arguments:
            file_in_name:  a path where we read the non-sequence formatted file.
            file_out_name: the path of the output data file. If not set, it would
                           be configured as file_in_name+'_seq'
        '''
        self.f_in = None
        self.f_out = None
        self.__in_context = False
        self.__kwargs_ = {
            'logver': 0,
            'seq_len': 10,
            'seq_len_max': 20,
            'set_shuffle': False,
            'random_seed': 2048
        }
        self.__kwargs = dict()
        if file_in_name is not None and file_in_name != '':
            self.open(file_in_name=file_in_name, file_out_name=file_out_name)
        self.config(dtype=np.float32)

    def __enter__(self):
        if self.f_in is None or self.f_out is None:
            raise FileNotFoundError('data.h5py: The saver is closed now, should open a new file before entering the contex.')
        self.__in_context = True
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.__close()
        self.__in_context = False

    def __config_pass(self, kwargs, keys):
        if not isinstance(keys, (list, tuple)):
            keys = (keys, )
        for k in keys:
            v = kwargs.pop(k, None)
            if v is not None:
                self.__kwargs_[k] = v
        return kwargs

    def config(self, **kwargs):
        '''
        Make configuration for the converter.
        Arguments for this class:
            logver (int): the log level for dumping files.
        Arguments often used:
            chunks (tuple):         size of data blocks.
            compression (str):      compression method.
            compression_opts (int): compression parameter.
            shuffle (bool):         shuffle filter for data compression.
            fletcher32 (bool):      check sum for chunks.
        Learn more available arguments here:
            http://docs.h5py.org/en/latest/high/dataset.html
        '''
        kwargs = self.__config_pass(kwargs, ('logver', 'set_shuffle', 'seq_len', 'seq_len_max', 'random_seed'))
        self.__kwargs.update(kwargs)
        if self.__kwargs_['logver'] > 0:
            print('data.h5py: Current configuration is:', self.__kwargs)

    def __generate_seqs(self, seq_len, seq_len_max, batch_size):
        '''Generate the subsequences.
        If the sub sequences does not exist, generate it. Otherwise, get it.
        '''
        if '$seqs' not in self.f_out:
            if batch_size < seq_len:
                raise ValueError('data.h5py: The number of mini-batches should not be smaller than the target sequence length.')
            if seq_len > seq_len_max:
                raise ValueError('data.h5py: The maximal length of the sequence should not be smaller than the sequence length.')
            if seq_len < 1:
                raise ValueError('data.h5py: The sequence length should be at least 1.')
            cur_size = batch_size
            try:
                rng = np.random.default_rng(seed=self.__kwargs_['random_seed'])
            except AttributeError:
                rng = np.random.RandomState(seed=self.__kwargs_['random_seed'])
            # Generate the sequences randomly.
            seqs = list()
            pre_sequence = 0
            while cur_size > 0:
                new_sequence = rng.integers(low=seq_len, high=min(seq_len_max + 1, cur_size))
                if cur_size - new_sequence < seq_len:
                    new_sequence = cur_size
                seqs.append((pre_sequence, pre_sequence + new_sequence))
                pre_sequence += new_sequence
                cur_size -= new_sequence
            seqs_ind = np.zeros((batch_size, 3), dtype=np.int)
            for i, (seq_begin, seq_end) in enumerate(seqs):
                seqs_ind[seq_begin:seq_end, 0] = i
                seqs_ind[seq_begin:seq_end, 1] = np.arange(start=0, stop=seq_end - seq_begin, dtype=np.int)
                seqs_ind[seq_begin:seq_end, 2] = seq_end - seq_begin
            seqs = np.asarray(seqs, dtype=np.int)
            dset = self.f_out.create_dataset('$seqs', data=seqs, maxshape=(None, 2), chunks=(1, 2), shuffle=True)
            dset.attrs['stype'] = 'system'
            dset = self.f_out.create_dataset('$seqs_ind', data=seqs_ind, maxshape=(None, 3), chunks=(1, 3), shuffle=True)
            dset.attrs['stype'] = 'system'
            if self.__kwargs_['logver'] > 1:
                print('data.h5py: Successfully generate the sequence indices: {0}'.format(seqs))
        else:
            seqs = self.f_out['$seqs'][:]
        return seqs

    def convert(self, keyword, **kwargs):
        '''
        Convert a keyword into sequence.
        The data would be converted into sequence. Note that before the
        conversion, the data should be arranged continuously of the
        batch axis.
        Arguments:
            keyword: the keyword that needs to be converted.
        Providing more configurations for `create_dataset` would override
        the default configuration defined by self.config().
        If you have already converted or copied the keyword, please do not
        do it again.
        '''
        if self.f_in is None or self.f_out is None:
            raise OSError('data.h5py: Should not copy data before opening a file.')
        if keyword not in self.f_in:
            raise KeyError('data.h5py: The required keyword {0} does not exist in the converted file.'.format(keyword))
        newkw = self.__kwargs.copy()
        newkw.update(kwargs)
        set_in = self.f_in[keyword]
        dshape = set_in.shape
        chunks = newkw.pop('chunks', None)
        if chunks is None:
            chunks = (1, *dshape[1:])
        seq_len = self.__kwargs_['seq_len']
        seq_len_max = self.__kwargs_['seq_len_max']
        set_shuffle = self.__kwargs_['set_shuffle']
        seqs = self.__generate_seqs(seq_len, seq_len_max, dshape[0])
        set_out = self.f_out.require_group(keyword)
        seq_indicies = np.arange(len(seqs), dtype=np.int)
        if set_shuffle:
            try:
                rng = np.random.default_rng(seed=self.__kwargs_['random_seed'])
            except AttributeError:
                rng = np.random.RandomState(seed=self.__kwargs_['random_seed'])
            rng.shuffle(seq_indicies)

        set_out['$seqs_ind'] = h5py.SoftLink('/$seqs_ind')
        for i, idx in enumerate(seq_indicies):
            seq_begin, seq_end = seqs[idx]
            set_out.create_dataset(str(i), data=set_in[seq_begin:seq_end, ...], maxshape=(None, *dshape[1:]), chunks=chunks, **newkw)
            if self.__kwargs_['logver'] > 1:
                print('data.h5py: Dump {0}: patch number={1}, patch length={2}'.format(keyword, i, seq_end - seq_begin))
        set_out.attrs['stype'] = 'sequence'

        if self.__kwargs_['logver'] > 0:
            print('data.h5py: Convert {0} into the output file. The original data shape is {1}, splitted into {2} parts.'.format(keyword, dshape, len(seqs)))

    def copy(self, keyword, **kwargs):
        '''
        Copy a keyword into sequence.
        The data belonging to the keyword would be copied into as it is
        directly.
        Arguments:
            keyword: the keyword that needs to be converted.
        Providing more configurations for `create_dataset` would override
        the default configuration defined by self.config().
        If you have already converted or copied the keyword, please do not
        do it again.
        '''
        if self.f_in is None or self.f_out is None:
            raise OSError('data.h5py: Should not copy data before opening a file.')
        if keyword not in self.f_in:
            raise KeyError('data.h5py: The required keyword {0} does not exist in the converted file.'.format(keyword))
        newkw = self.__kwargs.copy()
        newkw.update(kwargs)
        set_in = self.f_in[keyword]
        dshape = set_in.shape
        chunks = newkw.pop('chunks', None)
        if chunks is None:
            chunks = (1, *dshape[1:])
        if keyword in self.f_out:
            self.f_out[keyword][:] = set_in
        else:
            self.f_out.create_dataset(keyword, data=set_in, maxshape=(None, *dshape[1:]), chunks=chunks, **newkw)
        self.f_out[keyword].attrs['stype'] = 'batch'
        if self.__kwargs_['logver'] > 0:
            print('data.h5py: Copy {0} into the output file. The data shape is {1}.'.format(keyword, dshape))

    def open(self, file_in_name, file_out_name=None):
        '''
        The dumped file name (path), it will produce a .h5 file.
        Arguments:
            file_in_name:  a path where we read the non-sequence formatted file.
            file_out_name: the path of the output data file. If not set, it would
                           be configured as file_in_name+'_seq'
        '''
        if self.__in_context:
            raise RuntimeError('data.h5py: Should not open a file when the saver is managing a context, because there is already an opened file. Try to exit the context or create a new different saver.')
        file_in_name, file_in_ext = os.path.splitext(file_in_name)
        if file_out_name is None:
            file_out_name = file_in_name + '_seq'
        file_out_name, file_out_ext = os.path.splitext(file_out_name)
        if file_in_ext != '.h5':
            file_in_name += '.h5'
        if file_out_ext != '.h5':
            file_out_name += '.h5'
        self.close()
        self.f_in = h5py.File(file_in_name, 'r')
        self.f_out = h5py.File(file_out_name, 'w')
        if self.__kwargs_['logver'] > 0:
            print('data.h5py: Open a new read file:', file_in_name)
            print('data.h5py: Open a new output file:', file_out_name)
        return self

    def __close(self):
        if self.f_in is not None:
            self.f_in.close()
        self.f_in = None
        if self.f_out is not None:
            self.f_out.close()
        self.f_out = None

    def close(self):
        '''Close the converter.'''
        if self.__in_context:
            raise RuntimeError('data.h5py: Should not close the file explicitly when the saver is managing a context. Try to exit the context or create a new different saver.')
        self.__close()


class H5SupSaverGroup:
    '''A handle for wrapping the h5py.Group. This handle is also used
    for designing the nestable H5SupSaver file handle.
    '''
    def __init__(self, h, parent=None):
        '''
        Initialization. An example of using this handle is:
        ```python
            f = h5py.File(...)
            g = H5SupSaverGroup(f['...'])
        ```
        However, we do not recommend to use this handle by this way.
        This handle should be maintained by H5SupSaver automatically.
        Arguments:
            h: An h5py.File, h5py.Group or h5py.Dataset
        '''
        if isinstance(h, (h5py.File, h5py.Group, h5py.Dataset)):
            self.h = h
        else:
            raise TypeError('data.h5py: The handle "h" should be an h5py Object.')
        self.parent = parent
        self.__root = self.__get_root_saver()
        self.__root_valid = isinstance(self.__root, H5SupSaver)
        self.__in_context = False
        if self.__root_valid:
            self.__kwargs = self.__root.get_config('$root')
            self.__kwargs_ = self.__root.get_config('$root_')
        else:
            self.__kwargs = None
            self.__kwargs_ = None

    def __enter__(self):
        if not self.is_open:
            raise FileNotFoundError('data.h5py: The group is closed now, should not enter a new context with it anymore.')
        self.__in_context = True
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.__close()
        self.__in_context = False

    def __getitem__(self, keyword):
        if not self.is_open:
            raise OSError('data.h5py: The group is closed now, should not nest it anymore.')
        if keyword not in self.h:
            g = self.h.require_group(keyword)
            return H5SupSaverGroup(h=g, parent=self)
        else:
            g = self.h[keyword]
            if isinstance(g, h5py.Dataset):
                return g
            else:
                return H5SupSaverGroup(h=g, parent=self)

    def __setitem__(self, keyword, value):
        if not self.is_open:
            raise OSError('data.h5py: The group is closed now, should not nest it anymore.')
        if keyword not in self.h:
            self.h[keyword] = value
        else:
            raise OSError('data.h5py: The keyword "{0}" has existed in the group. Should not set it.')

    def __get_root_saver(self):
        parent = self.parent
        while isinstance(parent, H5SupSaverGroup):
            parent = parent.parent
        return parent

    def __close(self):
        if self.h is not None:
            if isinstance(self.h, h5py.File):
                self.h.close()
        self.h = None

    def close(self):
        if self.__in_context:
            raise RuntimeError('data.h5py: Should not close the group explicitly when the group is managing a context. Try to exit the context or create a new different group.')
        self.__close()

    @property
    def is_open(self):
        '''
        Check whether the root file or parent nodes closed.
        '''
        if self.h is None:
            return False
        if self.__root_valid and self.__root.f is None:
            self.__close()
            return False
        parent = self.parent
        while isinstance(parent, H5SupSaverGroup):
            if parent.h is None:
                self.__close()
                return False
            parent = parent.parent
        return True

    @property
    def attrs(self):
        return self.h.attrs

    @attrs.setter
    def attrs(self, new_attrs):
        if not isinstance(new_attrs, dict):
            raise TypeError('data.h5py: The attrs used for updating the group.attrs should be a dict.')
        h_attrs = self.h.attrs
        for k, v in new_attrs.items():
            h_attrs[k] = v
        if self.__kwargs_['logver'] > 1:
            print('data.h5py: Save attrs to "{0}", containing "{1}".'.format(self.h.name, ', '.join(new_attrs.keys())))

    def config(self, **kwargs):
        '''
        Make configuration for the root saver.
        This method would be applied to the root saver of this group
        directly. See H5SupSaver.config to find how to use it.
        '''
        if self.__root_valid:
            self.__root.config(**kwargs)

    def get_config(self, name):
        '''Get the config value.
        Arguments:
            name: the name of the required config value.
        Returns:
            1. the required config value.
        '''
        if self.__root_valid:
            return self.__root.get_config(name)
        else:
            raise IndexError('data.h5py: The root H5SupSaver does not exist.')

    def dump(self, keyword, data, **kwargs):
        '''
        Dump the dataset with a keyword into the file.
        Arguments:
            keyword: the keyword of the dumped dataset.
            data:    dataset, should be a numpy array.
        Providing more configurations for `create_dataset` would override
        the default configuration defined by self.config()
        If the provided `keyword` exists, the dataset would be resized for
        accepting more data.
        '''
        if not self.is_open:
            raise OSError('data.h5py: This group has been closed. Should not dump data with it anymore. Please try to open a new file or create a new group.')
        newkw = self.__kwargs.copy()
        newkw.update(kwargs)
        if isinstance(data, (int, float)):
            data = np.asarray([data])
        elif isinstance(data, (list, tuple)):
            data = np.asarray(data)
        dshape = data.shape[1:]
        if keyword in self.h:
            ds = self.h[keyword]
            if not isinstance(ds, h5py.Dataset):
                raise TypeError('data.h5py: The existed keyword "{0}" is not an h5py.Dataset. Could not dump data into it.'.format(keyword))
            dsshape = ds.shape[1:]
            if np.all(np.array(dshape, dtype=np.int) == np.array(dsshape, dtype=np.int)):
                N = len(ds)
                newN = data.shape[0]
                ds.resize(N + newN, axis=0)
                ds[N:N + newN, ...] = data
                if self.__kwargs_['logver'] > 0:
                    print('data.h5py: Dump {smp} data samples into the existed dataset {ds}. The data shape is {sze} now.'.format(smp=newN, ds=ds.name, sze=ds.shape))
            else:
                raise ValueError('data.h5py: The data set shape {0} does not match the input shape {1}.'.format(dsshape, dshape))
        else:
            chunks = newkw.pop('chunks', None)
            if chunks is None:
                chunks = (1, *dshape)
            self.h.create_dataset(keyword, data=data, maxshape=(None, *dshape), chunks=chunks, **newkw)
            if self.__kwargs_['logver'] > 0:
                print('data.h5py: Dump {0} into the file. The data shape is {1}.'.format(keyword, data.shape))

    def set_link(self, keyword, target, overwrite=True):
        '''
        Create a h5py.Softlink.
        Arguments:
            keyword:   the keyword of the soft link.
            target:    the reference of the soft link.
            overwrite: if not set true, would skip this step when the
                       the keyword exists.
        '''
        if not self.is_open:
            raise OSError('data.h5py: This group has been closed. Should not set links with it.')
        if keyword in self.h:
            if not overwrite:
                return
            # get_link = self.h.get(keyword, getlink=True)
            del self.h[keyword]
            self.h[keyword] = h5py.SoftLink(target)
            if self.__kwargs_['logver'] > 0:
                print('data.h5py: Delete the existed data "{0}", make it pointting to "{1}" now.'.format(keyword, target))
        else:
            self.h[keyword] = h5py.SoftLink(target)
            if self.__kwargs_['logver'] > 0:
                print('data.h5py: Create a soft link "{0}", pointting to "{1}".'.format(keyword, target))

    def set_attrs(self, keyword, attrs=None, **kwargs):
        '''
        Set attrs for an existed data group or dataset.
        Arguments:
            keyword: the keyword of the soft link.
            attrs:   the attributes those would be set.
            kwargs: more attributes those would be combined with
                    attrs by .update().
        '''
        if not self.is_open:
            raise OSError('data.h5py: This group has been closed. Should not set attributes with it. Please try to open a new file or create a new group.')
        if keyword not in self.h:
            raise KeyError('data.h5py: Could not find the keyword "{0}" for setting the attrs.'.format(keyword))
        if attrs is None:
            attrs = dict()
        attrs.update(kwargs)
        name = self.h[keyword].name
        g_attrs = self.h[keyword].attrs
        for k, v in attrs.items():
            if isinstance(v, dict):
                v = '$jsondict:' + json.dumps(v)
            g_attrs[k] = v
        if self.__kwargs_['logver'] > 0:
            print('data.h5py: Save attrs to "{0}", containing "{1}".'.format(name, ', '.join(attrs.keys())))

    def set_virtual_set(self, keyword, sub_set_keys, fill_value=0.0):
        '''
        Create a virtual dataset based on a list of subsets. All subsets
        require to be h5py.Dataset.
        Arguments:
            keyword:      the keyword of the newly created virtual
                          dataset.
            sub_set_keys: a list of sub-set keywords. Each sub-set
                          should share the same shape (except for the
                          first dimension).
            fill_value:   value used for filling the blank area in the
                          virtual dataset.
        '''
        if not self.is_open:
            raise OSError('data.h5py: This group has been closed. Should not set virtual datasets with it. Please try to open a new file or create a new group.')
        ds_list = list()
        len_max = 0
        set_shape = None
        set_dtype = None
        for k in sub_set_keys:
            ds = self.h[k]
            if isinstance(ds, h5py.Dataset):
                # Check the set_dtype
                dtype = ds.dtype
                if set_dtype is None:
                    set_dtype = dtype
                # Check the set_shape
                dshape = ds.shape[1:]
                if len(ds.shape) == 0:
                    raise ValueError('data.h5py: The sub-set "{0}" has an empty shape, not allowed.'.format(k))
                if set_shape is None:
                    set_shape = dshape
                else:
                    if (len(dshape) != len(set_shape)) or (not functools.reduce(lambda x, y: (x and y[0] == y[1]), zip(dshape, set_shape), True)):
                        raise ValueError('data.h5py: The sub-set "{0}" has a bad shape "{1}", not equaling to "{2}".'.format(k, dshape, set_shape))
                # Update the sample number.
                len_max = max(len_max, ds.shape[0])
                ds_list.append(ds)
            else:
                raise TypeError('data.h5py: The required keyword "{0}" is not a h5py.Dataset.'.format(k))
        if ds_list:
            n_d = len(ds_list)
            layout = h5py.VirtualLayout(shape=(len_max, n_d, *set_shape), dtype=set_dtype)
            for i, ds in enumerate(ds_list):
                vsource = h5py.VirtualSource(ds)
                layout[:len(ds), i, ...] = vsource
            vds = self.h.create_virtual_dataset(keyword, layout, fillvalue=fill_value)
            if self.__kwargs_['logver'] > 0:
                print('data.h5py: Create a new virtual dataset "{0}", containing "{1}" samples, "{2}" subsets.'.format(vds.name, len_max, n_d))


class H5SupSaver:
    '''Save supervised data set as .h5 file
    This class allows users to dump multiple datasets into one file
    handle, then it would save it as a .h5 file. The keywords of the
    sets should be assigned by users.
    '''
    def __init__(self, file_name=None, enable_read=False):
        '''
        Create the .h5 file while initialization.
        Arguments:
            file_name:   a path where we save the file.
            enable_read: when set True, enable the read/write mode.
                         This option is used when adding data to an
                         existed file.
        '''
        self.f = None
        self.__in_context = False
        self.__kwargs_ = {
            'logver': 0
        }
        self.__kwargs = dict()
        self.enable_read = enable_read
        if file_name is not None and file_name != '':
            self.open(file_name, enable_read=enable_read)
        self.config(dtype=np.float32)

    def __enter__(self):
        if self.f is None:
            raise FileNotFoundError('data.h5py: The saver is closed now, should open a new file before entering the context.')
        self.__in_context = True
        self.f.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.f.__exit__(exc_type, exc_value, exc_traceback)
        self.f = None
        self.__in_context = False

    def __config_pass(self, kwargs, keys):
        if not isinstance(keys, (list, tuple)):
            keys = (keys, )
        for k in keys:
            v = kwargs.pop(k, None)
            if v is not None:
                self.__kwargs_[k] = v
        return kwargs

    def get_config(self, name):
        '''Get the config value.
        Arguments:
            name: the name of the required config value.
        Returns:
            1. the required config value.
        '''
        if name == '$root':
            return self.__kwargs
        elif name == '$root_':
            return self.__kwargs_
        else:
            return self.__kwargs.get(name, self.__kwargs_[name])

    def config(self, **kwargs):
        '''
        Make configuration for the saver.
        Arguments for this class:
            logver (int): the log level for dumping files.
        Arguments often used:
            chunks (tuple):         size of data blocks.
            compression (str):      compression method.
            compression_opts (int): compression parameter.
            shuffle (bool):         shuffle filter for data compression.
            fletcher32 (bool):      check sum for chunks.
        Learn more available arguments here:
            http://docs.h5py.org/en/latest/high/dataset.html
        '''
        kwargs = self.__config_pass(kwargs, ('logver',))
        self.__kwargs.update(kwargs)
        if self.__kwargs_['logver'] > 0:
            print('data.h5py: Current configuration is:', self.__kwargs)

    def __getitem__(self, keyword):
        if self.f is None:
            raise OSError('data.h5py: Should not nest a group before opening a file.')
        return self.f[keyword]

    def __setitem__(self, keyword, value):
        if self.f is None:
            raise OSError('data.h5py: Should not nest a group before opening a file.')
        self.f[keyword] = value

    def open(self, file_name, enable_read=None):
        '''
        The dumped file name (path), it will produce a .h5 file.
        Arguments:
            file_name:   a path where we save the file.
            enable_read: when set True, enable the read/write mode.
                         This option is used when adding data to an
                         existed file. If set None, the "enable_read"
                         would be inherited from the class
                         definition. Otherwise, the class definition
                         "enable_read" would be updated by this new
                         value.
        '''
        if self.__in_context:
            raise RuntimeError('data.h5py: Should not open a file when the saver is managing a context, because there is already an opened file. Try to exit the context or create a new different saver.')
        file_name, file_ext = os.path.splitext(file_name)
        file_name += '.h5'
        self.close()
        if enable_read is None:
            enable_read = self.enable_read
        else:
            self.enable_read = enable_read
        if enable_read:
            fmode = 'a'
        else:
            fmode = 'w'
        self.f = H5SupSaverGroup(h5py.File(file_name, fmode), parent=self)
        if self.__kwargs_['logver'] > 0:
            print('data.h5py: Open a new file:', file_name)
        return self

    def close(self):
        '''Close the saver.'''
        if self.__in_context:
            raise RuntimeError('data.h5py: Should not close the file explicitly when the saver is managing a context. Try to exit the context or create a new different saver.')
        if self.f is not None:
            self.f.close()
        self.f = None

    @property
    def attrs(self):
        if self.f is None:
            raise OSError('data.h5py: Should not get attributes before opening a file.')
        return self.f.attrs

    @attrs.setter
    def attrs(self, new_attrs):
        if self.f is None:
            raise OSError('data.h5py: Should not update attributes before opening a file.')
        if not isinstance(new_attrs, dict):
            raise TypeError('data.h5py: The attrs used for updating the group.attrs should be a dict.')
        self.f.attrs = new_attrs

    def dump(self, keyword, data, **kwargs):
        '''
        Dump the dataset with a keyword into the file.
        Arguments:
            keyword: the keyword of the dumped dataset.
            data:    dataset, should be a numpy array.
        Providing more configurations for `create_dataset` would override
        the default configuration defined by self.config()
        If the provided `keyword` exists, the dataset would be resized for
        accepting more data.
        '''
        if self.f is None:
            raise OSError('data.h5py: Should not dump data before opening a file.')
        self.f.dump(keyword, data, **kwargs)

    def set_link(self, keyword, target, overwrite=True):
        '''
        Create a h5py.Softlink.
        Arguments:
            keyword:   the keyword of the soft link.
            target:    the reference of the soft link.
            overwrite: if not set true, would skip this step when the
                       the keyword exists.
        '''
        if self.f is None:
            raise OSError('data.h5py: Should not set links before opening a file.')
        self.f.set_link(keyword, target, overwrite=overwrite)

    def set_attrs(self, keyword, attrs=None, **kwargs):
        '''
        Set attrs for an existed data group or dataset.
        Arguments:
            keyword: the keyword where we set the attributes.
            attrs:   the attributes those would be set.
            kwargs: more attributes those would be combined with
                    attrs by .update().
        '''
        if self.f is None:
            raise OSError('data.h5py: Should not set attributes before opening a file.')
        self.f.set_attrs(keyword, attrs=attrs, **kwargs)

    def set_virtual_set(self, keyword, sub_set_keys, fill_value=0.0):
        '''
        Create a virtual dataset based on a list of subsets. All subsets
        require to be h5py.Dataset.
        Arguments:
            keyword:      the keyword of the newly created virtual
                          dataset.
            sub_set_keys: a list of sub-set keywords. Each sub-set
                          should share the same shape (except for the
                          first dimension).
            fill_value:   value used for filling the blank area in the
                          virtual dataset.
        '''
        if self.f is None:
            raise OSError('data.h5py: Should not set virtual datasets before opening a file.')
        self.f.set_virtual_set(keyword, sub_set_keys, fill_value=fill_value)


class _H5AParser(abc.ABC):
    '''Abstract prototype of the H5*Parsers.
    Should not be directly used by users.
    '''

    @abc.abstractmethod
    def __init__(self):
        '''The base class method, require to be implemented.'''
        self.file_path = None
        self.batch_num = 0
        self.smanager = tuple()
        self.preproc = self._preproc

    @staticmethod
    def _preproc(x):
        return x

    @staticmethod
    def __get_dset_len(dset):
        if 'len' in dset.attrs:
            return dset.attrs['len']
        if isinstance(dset, h5py.Dataset):
            return len(dset)
        else:
            get_len = 0
            for d in dset:
                if not d.startswith('$'):
                    get_len += len(dset[d])
            return get_len

    def check_dsets(self, file_path, keywords):
        '''Check the size of datasets and validate all datasets.
        If success, would return the size of the datasets.
        Arguments:
            file_path: the path of the HDF5 dataset to be validated.
            keywords: the keywords to be validated. Each keyword should
                      point to or redict to a dataset.
        Returns:
            1. the size of all datasets.
        '''
        with h5py.File(file_path, 'r') as f:
            dsets = []
            for key in keywords:
                dsets.append(f[key])
            if not dsets:
                raise KeyError('data.h5py: Keywords are not mapped to datasets in the file.')
            sze = self.__get_dset_len(dsets[0])
            for dset in dsets:
                if sze != self.__get_dset_len(dset):
                    raise TypeError('data.h5py: The assigned keywords do not correspond to each other.')
        return sze

    def get_attrs(self, keyword, *args, attr_names=None):
        '''Get the attributes by the keyword.
        Arguments:
            keyword: the keyword of a dataset.
            attr_names: a sequence of required attribute names.
            *args: other attribute names, would be attached to the argument
                   "attr_names".
        Returns:
            1. a list of the required attribute values.
        '''
        if attr_names is None:
            attr_names = list()
        elif not isinstance(attr_names, (list, tuple)):
            attr_names = [str(attr_names)]
        else:
            attr_names = list(attr_names)
        attr_names.extend(args)
        with h5py.File(self.file_path, 'r') as f:
            attrs = f[keyword].attrs
            res = list()
            for n in attr_names:
                v = attrs[n]
                if isinstance(v, str) and v.startswith('$jsondict:'):
                    v = json.loads(v[10:])
                res.append(v)
        return res

    def get_file(self, enable_write=False):
        '''Get a file object of the to-be-loaded file.
        Arguments:
            enable_write: if enabled, would use the `a` mode to open the file.
                          Otherwise, use the `r` mode.
        Returns:
            1. the h5py.File object of the to-be-loaded file.
        '''
        mode = 'a' if enable_write else 'r'
        return h5py.File(self.file_path, mode)

    def __len__(self):
        '''
        Automatically calculate the steps for iterate the whole dataset.
        '''
        return self.batch_num

    def __enter__(self):
        self.smanager.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.smanager.__exit__(exc_type, exc_value, exc_traceback)

    def start(self, compat=None):
        '''Start the process pool.
        Could be used like:
        ```python
            parser.start()
            for ... in parser:
                ...
            parser.finish()
        ```
        Or using the context:
        ```python
            with parser.start() as m:
                for ... in m:
                    ...
        ```
        The cuda.Tensor could not be put into the queue on Windows (but on Linux we could), see
            https://pytorch.org/docs/stable/notes/windows.html#cuda-ipc-operations
        To solve this problem, we need to fall back to multi-threading for converter on Windows.
        Argument:
            compat: whether to fall back to multi-threading for the converter. If set None, the
                decision would be made by checking os.name.
        '''
        self.smanager.start(compat=compat)
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
        self.smanager.start_test(test_mode=test_mode)
        return self

    def finish(self):
        '''Finish the process pool.
        The compatible mode would be auto detected by the previous start().
        '''
        self.smanager.finish()

    def __iter__(self):
        return iter(self.smanager)


class _H5GParser:
    def __init__(self, file_name, keywords, preprocfunc=None):
        self.f = None
        self.keywords = keywords
        if (not os.path.isfile(file_name)) and (os.path.isfile(file_name + '.h5')):
            file_name += '.h5'
        self.f = h5py.File(file_name, 'r')
        self.__dsets = self.create_datasets()
        self.__dsize = len(self.__dsets)
        self.__preprocfunc = preprocfunc

    @staticmethod
    def __get_item_dset(dset, index):
        return dset[index]

    @staticmethod
    def __get_item_seq(dset, index):
        set_ind, samp_ind = dset['$seqs_ind'][index, :2]
        return dset[str(set_ind)][samp_ind]

    def create_datasets(self):
        '''
        Find all desired dataset handles, and store them.
        '''
        dsets = []
        for key in self.keywords:
            dset = self.f[key]
            if isinstance(dset, h5py.Dataset):
                dsets.append((dset, self.__get_item_dset))
            else:
                dsets.append((dset, self.__get_item_seq))
        if not dsets:
            raise KeyError('data.h5py: Keywords are not mapped to datasets in the file.')
        return dsets

    def __getitem__(self, batch_indices):
        res = []
        for j in range(self.__dsize):
            res.append([])
        for ind in batch_indices:
            smp = self.__map_single(ind)
            for j in range(self.__dsize):
                res[j].append(smp[j])
        for j in range(self.__dsize):
            res[j] = np.stack(res[j], axis=0)
        if self.__preprocfunc is not None:
            return self.__preprocfunc(*res)
        else:
            return tuple(res)

    def __map_single(self, index):
        '''
        Map function, for multiple datasets mode.
        '''
        return tuple(get_item(dset, index) for (dset, get_item) in self.__dsets)


class _H5CParser:
    def __init__(self, file_name, keywords_sequence, keywords_single,
                 sequence_size, sequence_position, sequence_padding,
                 data_size, preprocfunc=None):
        self.f = None
        # Parse keywords.
        self.keywords_sequence = keywords_sequence
        self.keywords_single = keywords_single
        # Parse the options
        self.sequence_size = sequence_size
        self.sequence_position = sequence_position
        if sequence_padding == 'same':
            self.__get_seq_item = self.__get_seq_item_pad_same
            self.__map_single = self.__map_single_with_pad
        elif sequence_padding == 'zero':
            self.__get_seq_item = self.__get_seq_item_pad_zero
            self.__map_single = self.__map_single_with_pad
        else:
            self.__get_seq_item = self.__get_seq_item_pad_none
            self.__map_single = self.__map_single_no_pad
        self.data_size = data_size

        # Parse the file name.
        if (not os.path.isfile(file_name)) and (os.path.isfile(file_name + '.h5')):
            file_name += '.h5'
        self.f = h5py.File(file_name, 'r')

        # Parse the sequence sets.
        self.__dsets_seq = self.create_datasets(self.keywords_sequence, create_sequence=True)
        self.__dsize_seq = len(self.__dsets_seq)
        # Parse the single sets.
        self.__dsets_sig = self.create_datasets(self.keywords_single, create_sequence=False)
        self.__dsize_sig = len(self.__dsets_sig)

        self.__preprocfunc = preprocfunc

    def __calculate_indices(self, index, data_size):
        if index < self.sequence_position:
            left_repeat = self.sequence_position - index
            index_start = 0
        else:
            left_repeat = 0
            index_start = index - self.sequence_position
        if index - self.sequence_position + self.sequence_size > data_size:
            right_repeat = index - self.sequence_position + self.sequence_size - data_size
            index_end = data_size
        else:
            right_repeat = 0
            index_end = index - self.sequence_position + self.sequence_size
        return index_start, index_end, left_repeat, right_repeat

    @staticmethod
    def __get_single_item_dset(dset, index):
        return dset[index]

    @staticmethod
    def __get_single_item_seq(dset, index):
        set_ind, samp_ind = dset['$seqs_ind'][index, :2]
        return dset[str(set_ind)][samp_ind]

    def __get_true_dset(self, dset_group, index, is_seq=False):
        if is_seq:
            set_ind, samp_ind, set_len = dset_group['$seqs_ind'][index]
            dset = dset_group[str(set_ind)]
            return dset, samp_ind, set_len
        else:
            return dset_group, index, self.data_size

    def __get_seq_item_pad_same(self, dset, index, is_seq=False):
        dset, index, data_size = self.__get_true_dset(dset, index, is_seq=is_seq)
        index_start, index_end, left_repeat, right_repeat = self.__calculate_indices(index, data_size)
        dset_item = list()
        if left_repeat > 0:
            dset_item.append(np.repeat(np.expand_dims(dset[0], axis=0), left_repeat, axis=0))
        if index_start < index_end:
            dset_item.append(dset[index_start:index_end])
        if right_repeat > 0:
            dset_item.append(np.repeat(np.expand_dims(dset[-1], axis=0), right_repeat, axis=0))
        return np.concatenate(dset_item, axis=0)

    def __get_seq_item_pad_zero(self, dset, index, is_seq=False):
        dset, index, data_size = self.__get_true_dset(dset, index, is_seq=is_seq)
        index_start, index_end, left_repeat, right_repeat = self.__calculate_indices(index, data_size)
        dset_item = list()
        dshape = dset.shape[1:]
        if left_repeat > 0:
            dset_item.append(np.zeros([left_repeat, *dshape]))
        if index_start < index_end:
            dset_item.append(dset[index_start:index_end])
        if right_repeat > 0:
            dset_item.append(np.zeros([right_repeat, *dshape]))
        return np.concatenate(dset_item, axis=0)

    def __get_seq_item_pad_none(self, dset, index, is_seq=False):
        dset, index, data_size = self.__get_true_dset(dset, index, is_seq=is_seq)
        index_start = index
        index_end = index + self.sequence_size
        return dset[index_start:index_end]

    def create_datasets(self, keywords, create_sequence=False):
        '''
        Find all desired dataset handles, and store them.
        '''
        if create_sequence:
            __get_item_dset = functools.partial(self.__get_seq_item, is_seq=False)
            __get_item_seq = functools.partial(self.__get_seq_item, is_seq=True)
        else:
            __get_item_dset = self.__get_single_item_dset
            __get_item_seq = self.__get_single_item_seq
        dsets = []
        for key in keywords:
            dset = self.f[key]
            if isinstance(dset, h5py.Dataset):
                dsets.append((dset, __get_item_dset))
            else:
                dsets.append((dset, __get_item_seq))
        return dsets

    def __getitem__(self, batch_indices):
        # Get seq results.
        res_seq = []
        for j in range(self.__dsize_seq):
            res_seq.append([])
        if self.__dsize_seq > 0:
            for ind in batch_indices:
                smp = self.__map_sequence_pad(ind)
                for j in range(self.__dsize_seq):
                    res_seq[j].append(smp[j])
            for j in range(self.__dsize_seq):
                res_seq[j] = np.stack(res_seq[j], axis=0)
        # Get single results.
        res_sig = []
        for j in range(self.__dsize_sig):
            res_sig.append([])
        if self.__dsize_sig > 0:
            for ind in batch_indices:
                smp = self.__map_single(ind)
                for j in range(self.__dsize_sig):
                    res_sig[j].append(smp[j])
            for j in range(self.__dsize_sig):
                res_sig[j] = np.stack(res_sig[j], axis=0)
        # Combine results
        res = res_seq + res_sig
        if self.__preprocfunc is not None:
            return self.__preprocfunc(*res)
        else:
            return tuple(res)

    def __map_sequence_pad(self, index):
        return tuple(get_item(dset, index) for (dset, get_item) in self.__dsets_seq)

    def __map_single_with_pad(self, index):
        return tuple(get_item(dset, index) for (dset, get_item) in self.__dsets_sig)

    def __map_single_no_pad(self, index):
        return tuple(get_item(dset, index + self.sequence_position) for (dset, get_item) in self.__dsets_sig)


class _H5RParser:
    def __init__(self, file_name, keywords, procfunc=None):
        self.f = None
        if isinstance(keywords, str):
            self.keywords = (keywords,)
        else:
            self.keywords = keywords
        if (not os.path.isfile(file_name)) and (os.path.isfile(file_name + '.h5')):
            file_name += '.h5'
        self.f = h5py.File(file_name, 'r')
        self.__dsets = self.create_datasets()
        self.__dsize = len(self.__dsets)
        self.__procfunc = procfunc

    def create_datasets(self):
        '''
        Find all desired dataset handles, and store them.
        '''
        dsets = []
        for key in self.keywords:
            dsets.append(self.f[key])
        if not dsets:
            raise KeyError('data.h5py: Keywords are not mapped to datasets in the file.')
        return dsets

    def __getitem__(self, batch_indices):
        return self.__procfunc(*self.__dsets)

    def __map_single(self, index):
        '''
        Map function, for multiple datasets mode.
        '''
        return tuple(dset[index] for dset in self.__dsets)


class H5GParser(_H5AParser):
    '''Grouply parsing dataset
    This class allows users to feed one .h5 file, and convert it to
    data.sequence.MP(T)Sequence. The realization could be described as:
        (1) Create `.h5` file indexer, this indexer would be
            initialized by sequence.MPSequence. It would use the
            user defined keywords to get a group of datasets.
        (2) Estimate the dataset sizes, each dataset should share the
            same size (but could have different shapes).
        (3) Use the dataset size to create a sequence.MPSequence, and
            allows it to randomly shuffle the indices in each epoch.
        (4) Invoke the MPSequence APIs to serve the parallel dataset
            parsing.
    Certainly, you could use this parser to load a single dataset.
    '''
    def __init__(self, file_name, keywords, batch_size=32, shuffle=True, shuffle_seed=1000, preprocfunc=None, thread_type='proc', num_workers=4, num_buffer=10):
        '''
        Create the parser and its h5py file handle.
        Arguments:
            file_name: the data path of the file (could be without postfix).
            keywords: should be a list of keywords (or a single keyword).
            batch_size: number of samples in each batch.
            shuffle: if on, shuffle the data set at the beginning of each
                     epoch.
            shuffle_seed: the seed for random shuffling.
            preprocfunc: this function would be added to the produced data
                         so that it could serve as a pre-processing tool.
                         Note that this tool would process the batches
                         produced by the parser.
            thread_type: the backend of the multi-threading, could be 'proc' or
                         'thread'.
            num_workers: the number of workers.
            num_buffer: the buffer size of the data pool, it means the number
                        of mini-batches.
        '''
        if isinstance(keywords, str):
            self.keywords = (keywords,)
        elif isinstance(keywords, (list, tuple)):
            self.keywords = keywords
        else:
            raise TypeError('data.h5py: The keywords should be a list of str or a str.')
        if (not isinstance(thread_type, str)) or thread_type not in ('proc', 'thread'):
            raise TypeError('data.h5py: The argument "thread_type" requires to be "proc" or "thread".')
        if len(self.keywords) == 0 or any(map(lambda x: (not isinstance(x, str)), self.keywords)):
            raise TypeError('data.h5py: The keywords should be a list of str or a str.')
        if (not os.path.isfile(file_name)) and (os.path.isfile(file_name + '.h5')):
            self.file_path = file_name + '.h5'
        else:
            self.file_path = file_name
        self.size = self.check_dsets(self.file_path, self.keywords)
        self.batch_size = batch_size
        self.batch_num = np.ceil(self.size / batch_size).astype(np.int)

        self.worker = functools.partial(_H5GParser, file_name=file_name, keywords=self.keywords, preprocfunc=preprocfunc)
        self.preproc = preprocfunc if preprocfunc is not None else super()._preproc
        Manager = sequence.MPSequence if thread_type == 'proc' else sequence.MTSequence
        self.smanager = Manager(self.worker, dset_size=self.size, num_workers=num_workers, num_converters=1, batch_size=batch_size, buffer=num_buffer, shuffle=shuffle, seed=shuffle_seed)


class H5CParser(_H5AParser):
    '''Continuously parsing dataset
    This class allows users to feed one .h5 file, and convert it to
    data.sequence.MP(T)Sequence. The realization could be described as:
    This Parser is the upgraded version of H5GParser, it is specially
    designed for parsing data to LSTM/ConvLSTM. A `sequence` dimension
    would be inserted between `batches` and `channels`. In each batch,
    the sequence is continuously extracted in the order of the batches.
    '''
    def __init__(self, file_name, keywords_sequence, keywords_single, batch_size=32,
                 sequence_size=5, sequence_position=-1, sequence_padding='same',
                 shuffle=True, shuffle_seed=1000, thread_type='proc', preprocfunc=None,
                 num_workers=4, num_buffer=10):
        '''
        Create the parser and its h5py file handle.
        Arguments:
            file_name: the data path of the file (could be without postfix).
            keywords_sequence: the keyword of sequence data. The keywords in
                               this list would be parsed as (B, S, C1, C2, ...).
                               It should be a list of keywords
                               (or a single keyword).
            keyword_single: the keyword of single values. The keywords in this
                            list would be parsed as (B, C1, C2, ...). It should
                            be a list of keywords (or a single keyword).
            batch_size: number of samples in each batch.
            sequence_size: the size of each sequence. It represents `S` of
                           (B, S, C1, C2, ...).
            sequence_position: the aligned position between the single values
                               and the sequence values. It should be in the
                               range of >= 0 and < 'sequence_size'.
            sequence_padding: the padding method for each epoch, it will
                              influence the first or the final samples in
                              the dataset. Could be 'same', 'zero' or
                              'none'. If None, the number of batches of each
                              epoch would be a little bit smaller than the
                              actual number.
            shuffle: if on, shuffle the data set at the end of each epoch.
            shuffle_seed: the seed for random shuffling.
            preprocfunc: this function would be added to the produced data
                         so that it could serve as a pre-processing tool.
                         Note that this tool would process the batches
                         produced by the parser.
            thread_type: the backend of the multi-threading, could be 'proc' or
                         'thread'.
            num_workers: the number of workers.
            num_buffer: the buffer size of the data pool, it means the number
                        of mini-batches.
        '''
        if isinstance(keywords_sequence, str):
            self.keywords_sequence = (keywords_sequence,)
        elif isinstance(keywords_sequence, (list, tuple)):
            self.keywords_sequence = keywords_sequence
        elif keywords_sequence is None:
            self.keywords_sequence = tuple()
        if isinstance(keywords_single, str):
            self.keywords_single = (keywords_single,)
        elif isinstance(keywords_single, (list, tuple)):
            self.keywords_single = keywords_single
        elif keywords_single is None:
            self.keywords_single = tuple()
        if (not isinstance(thread_type, str)) or thread_type not in ('proc', 'thread'):
            raise TypeError('data.h5py: The argument "thread_type" requires to be "proc" or "thread".')
        # Validate the keywords
        keywords = self.keywords_sequence + self.keywords_single
        if len(keywords) == 0 or any(map(lambda x: (not isinstance(x, str)), keywords)):
            raise TypeError('data.h5py: The keywords should be a list of str or a str.')

        if (not os.path.isfile(file_name)) and os.path.isfile(file_name + '.h5'):
            self.file_path = file_name + '.h5'
        else:
            self.file_path = file_name

        # Get padding configs.
        if sequence_size < 1:
            raise ValueError('data.h5py: The argument "sequence_size" should be a positive integer, but given {0}.'.format(sequence_size))
        self.sequence_size = sequence_size
        self.sequence_position = sequence_position % sequence_size
        if sequence_padding not in ('same', 'zero'):
            sequence_padding = None
        self.sequence_padding = sequence_padding

        # Check data size
        data_size = self.check_dsets(self.file_path, keywords)  # Get the actual size of the data
        if sequence_padding is None:
            mps_indices = self.get_indices(self.file_path, self.keywords_sequence, data_size)
            if isinstance(mps_indices, int):
                self.size = mps_indices
            else:
                self.size = len(mps_indices)
        else:
            mps_indices = data_size
            self.size = data_size
        self.batch_size = batch_size
        self.batch_num = np.ceil(self.size / batch_size).astype(np.int)
        if self.batch_num < num_workers:
            raise ValueError('data.h5py: The number of sequence samples are not enough, minimal required: {0}, but actually have {1}.'.format(num_workers * batch_size, self.size))

        self.worker = functools.partial(_H5CParser, file_name=file_name, keywords_sequence=self.keywords_sequence, keywords_single=self.keywords_single,
                                        sequence_size=self.sequence_size, sequence_position=self.sequence_position, sequence_padding=self.sequence_padding,
                                        data_size=data_size, preprocfunc=preprocfunc)
        self.preproc = preprocfunc if preprocfunc is not None else super()._preproc
        Manager = sequence.MPSequence if thread_type == 'proc' else sequence.MTSequence
        self.smanager = Manager(self.worker, dset_size=mps_indices, num_workers=num_workers, num_converters=1, batch_size=batch_size, buffer=num_buffer, shuffle=shuffle, seed=shuffle_seed)

    def get_indices(self, file_path, keywords, data_size):
        with h5py.File(file_path, 'r') as f:
            # Check whether using seq data.
            is_seq = False
            for key in keywords:
                if not isinstance(f[key], h5py.Dataset):
                    is_seq = True
                    break
            if is_seq:
                indicies = list()
                for seq_begin, seq_end in f['$seqs'][:]:
                    indicies.append(np.arange(start=seq_begin, stop=seq_end - self.sequence_size + 1, dtype=np.int))
                indicies = np.concatenate(indicies, axis=0)
        if is_seq:
            return indicies
        else:
            return data_size - self.sequence_size + 1


class H5RParser(_H5AParser):
    '''Randomly parsing dataset
    This class allows users to feed one .h5 file, and convert it to
    data.sequence.MPSequence. The realization could be described as:
        (1) Create .h5 file handle.
        (2) Using the user defined keywords to get a group of datasets.
        (3) Check the dataset size, and register the dataset list.
        (4) In each iteration, iterate all datasets.
    Certainly, you could use this parser to load a single dataset.
    '''
    def __init__(self, file_name, keywords, preprocfunc, batch_num=100, thread_type='proc', num_workers=4, num_buffer=10):
        '''
        Create the parser and its h5py file handle.
        Arguments:
            file_name: the data path of the file (could be without postfix).
            keywords: should be a list of keywords (or a single keyword).
            preprocfunc: this function would be added to the produced data
                         so that it could serve as a pre-processing tool.
            batch_num: the number of mini-batches in each epoch.
            batch_size: number of samples in each mini-batch.
            thread_type: the backend of the multi-threading, could be 'proc' or
                         'thread'.
            num_workers: the number of workers.
            num_buffer: the buffer size of the data pool, it means the number
                        of mini-batches.
        '''
        if isinstance(keywords, str):
            self.keywords = (keywords,)
        elif isinstance(keywords, (list, tuple)):
            self.keywords = keywords
        else:
            raise TypeError('data.h5py: The keywords should be a list of str or a str.')
        if (not isinstance(thread_type, str)) or thread_type not in ('proc', 'thread'):
            raise TypeError('data.h5py: The argument "thread_type" requires to be "proc" or "thread".')
        if len(self.keywords) == 0 or any(map(lambda x: (not isinstance(x, str)), self.keywords)):
            raise TypeError('data.h5py: The keywords should be a list of str or a str.')
        if (not os.path.isfile(file_name)) and (os.path.isfile(file_name + '.h5')):
            self.file_path = file_name + '.h5'
        else:
            self.file_path = file_name
        self.check_dsets(self.file_path, self.keywords)
        self.size = batch_num
        self.batch_num = batch_num

        self.worker = functools.partial(_H5RParser, file_name=file_name, keywords=keywords, procfunc=preprocfunc)
        self.preproc = preprocfunc if preprocfunc is not None else super()._preproc
        Manager = sequence.MPSequence if thread_type == 'proc' else sequence.MTSequence
        self.smanager = Manager(self.worker, dset_size=self.size, num_workers=num_workers, num_converters=1, batch_size=1, buffer=num_buffer, shuffle=False)
