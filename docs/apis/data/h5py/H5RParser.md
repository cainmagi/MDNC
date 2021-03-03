# data.h5py.H5RParser

:codicons-symbol-class: Class · [:octicons-file-code-24: Source]({{ source.root }}/data/h5py.py#L1469)

```python
dset = mdnc.data.h5py.H5RParser(
    file_name, keywords, preprocfunc, batch_num=100,
    num_workers=4, num_buffer=10
)
```

This class allows users to feed one .h5 file, and convert it to [`mdnc.data.sequence.MPSequence`](../sequence/MPSequence). The realization could be described as:

1. Create .h5 file handle.
2. Using the user defined keywords to get a group of datasets.
3. Check the dataset size, and register the dataset list.
4. For each batch, the data is randomly picked from the whole set. The `h5py.Dataset` variable would be transparent in the `preprocfunc`, i.e. the method how to pick up the random samples need to be implemented by users.

Certainly, you could use this parser to load a single dataset.

## Arguments

**Requries**

| Argument {: .w-6rem} | Type {: .w-5rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `file_name` | `#!py str` | The path of the `.h5` file (could be without postfix). |
| `keywords` | `#!py (str, )` | Should be a list of keywords (or a single keyword). |
| `preprocfunc` | `#!py object` | This function would be added to the produced data so that it could serve as a pre-processing tool. This function is required because the random sampling needs to be implemented here. |
| `batch_num` | `#!py int` | Number of mini-batches in each epoch. |
| `num_workers` | `#!py int` | The number of parallel workers. |
| `num_buffer` | `#!py int` | The buffer size of the data pool, it means the maximal number of mini-batches stored in the memory. |

??? tip
    The `preprocfunc` is required in this case. The provided pre-processors in `data.preprocs` should not be used directly, because users need to implment their own random sampling pre-processor first. For example,

    !!! example
        === "Without data.preprocs"
            ```python linenums="1"
            import numpy as np
            import mdnc

            class ProcCustom:
                def __init__(self, seed=1000, batch_size=16):
                    self.batch_size = batch_size
                    self.random_rng = np.random.default_rng(seed)

                def __call__(self, ds_x1, ds_x2):
                    ind_x1 = np.sort(self.random_rng.integers(len(ds_x1), size=batch_size))
                    ind_x2 = np.sort(self.random_rng.integers(len(ds_x2), size=batch_size))
                    return ds_x1[ind_x1, ...], ds_x2[ind_x2, ...]

            mdnc.data.h5py.H5RParser(..., preprocfunc=ProcCustom(), keywords=['x_1', 'x_2'])
            ```

        === "Use data.preprocs"
            ```python linenums="1"
            import numpy as np
            import mdnc

            class ProcCustom(mdnc.data.preprocs.ProcAbstract):
                def __init__(self, seed=1000, batch_size=16, inds=None, parent=None):
                    super().__init__(inds=inds, parent=parent)
                    self.batch_size = batch_size
                    self.random_rng = np.random.default_rng(seed)

                def preprocess(self, ds):
                    ind = np.sort(self.random_rng.integers(len(ds), size=batch_size))
                    return ds[ind, ...]

                def postprocess(self, x):
                    return x

            mdnc.data.h5py.H5RParser(..., keywords=['x_1', 'x_2'],
                                     preprocfunc=mdnc.data.preprocs.ProcScaler(parent=ProcCustom()))
            ```

    !!! warning
        The argument `preprocfunc` requires to be a [picklable object][pydoc-picklable]. Therefore, a lambda function or a function implemented inside `#!py if __name__ == '__main__'` is not allowed in this case.

## Methods

### :codicons-symbol-method: `check_dsets`

```python
sze = dset.check_dsets(file_path, keywords)
```

Check the size of `#!py h5py.Dataset` and validate all datasets. A valid group of datasets requires each `#!py h5py.Dataset` shares the same length (sample number). If success, would return the size of the datasets. This method is invoked during the initialization, and do not requires users to call explicitly.

**Requries**

| Argument {: .w-5rem} | Type {: .w-5rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `file_path` | `#!py str` | The path of the HDF5 dataset to be validated. |
| `keywords` | `#!py (str, )` | The keywords to be validated. Each keyword should point to or redict to an `#!py h5py.Dataset`. |

**Returns**

| Argument {: .w-5rem} | Description {: .w-8rem} |
| :------: | :---------- |
| `sze` | A `#!py int`, the size of all datasets. |

-----

### :codicons-symbol-method: `get_attrs`

```python
attrs = dset.get_attrs(keyword, *args, attr_names=None)
```

Get the attributes by the keyword.

**Requries**

| Argument {: .w-6rem} | Type {: .w-5rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `keyword` | `#!py str` | The keyword of the to a `h5py.Dataset` in the to-be-loaded file. |
| `attr_names` | `#!py (str, )` | A sequence of required attribute names. |
| `*args` | | other attribute names, would be attached to the argument `attr_names` by `#!py list.extend()`. |

**Returns**

| Argument {: .w-5rem} | Description {: .w-8rem} |
| :------: | :---------- |
| `attrs` | A list of the required attribute values. |

-----

### :codicons-symbol-method: `get_file`

```python
f = dset.get_file(enable_write=False)
```

Get a file object of the to-be-loaded file.

**Requries**

| Argument {: .w-6rem} | Type {: .w-4rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `enable_write` | `#!py bool` | If enabled, would use the `a` mode to open the file. Otherwise, use the `r` mode. |

**Returns**

| Argument {: .w-5rem} | Description {: .w-8rem} |
| :------: | :---------- |
| `f` | The `#!py h5py.File` object of the to-be-loaded file. |

-----

### :codicons-symbol-method: `start`

```python
dset.start(compat=None)
```

Start the process pool. This method is implemented by [`mdnc.data.sequence.MPSequence`](../sequence/MPSequence). It supports context management.

Running `start()` or `start_test()` would interrupt the started sequence.

**Requries**

| Argument {: .w-5rem} | Type {: .w-4rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `compat` | `#!py bool` | Whether to fall back to multi-threading for the sequence out-type converter. If set None, the decision would be made by checking `#!py os.name`. The compatible mode requires to be enabled on Windows. |

??? tip
    This method supports context management. Using the context is recommended. Here we show two examples:
    === "Without context"
        ```python linenums="1"
        dset.start()
        for ... in dset:
            ...
        dset.finish()
        ```

    === "With context"
        ```python linenums="1"
        with dset.start() as ds:
            for ... in ds:
                ...
        ```

??? danger
    The `#!py cuda.Tensor` could not be put into the queue on Windows (but on Linux we could), see

    https://pytorch.org/docs/stable/notes/windows.html#cuda-ipc-operations

    To solve this problem, we need to fall back to multi-threading for the sequence out-type converter on Windows.

??? warning
    Even if you set `#!py shuffle=False`, due to the mechanism of the parallelization, the sample order during the iteration may still get a little bit shuffled. To ensure your sample order not changed, please use `#!py shuffle=False` during the initialization and use [`#!py start_test()`](#start_test) instead.

-----

### :codicons-symbol-method: `start_test`

```python
dset.start_test(test_mode='default')
```

Start the test mode. In the test mode, the process pool would not be open. All operations would be finished in the main thread. However, the random indices are still generated with the same seed of the parallel `#!py dset.start()` mode.

Running `start()` or `start_test()` would interrupt the started sequence.

**Requries**

| Argument {: .w-6rem} | Type {: .w-4rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `test_mode` | `#!py str` | Could be `#!py 'default'`, `#!py 'cpu'`, or `#!py 'numpy'`. <ul> <li>`#!py 'default'`: the output would be converted as `start()` mode.</li> <li>`#!py 'cpu'`: even set 'cuda' as output type, the testing output would be still not converted to GPU.</li> <li>`#!py 'numpy'`: would ignore all out_type configurations and return the original output. This output is still pre-processed.</li> </ul>  |

??? tip
    This method also supports context management. See [`data.h5py.H5RParser.start`](#start) to check how to use it.

-----

### :codicons-symbol-method: `finish`

```python
dset.finish()
```

Finish the process pool. The compatible mode would be auto detected by the previous `start()`.

## Properties

### :codicons-symbol-property: `len()`, `batch_num`

```python
len(dset)
dset.batch_num
```

The length of the dataset. It is the number of mini-batches, also the number of iterations for each epoch.

-----

### :codicons-symbol-property: `iter()`

```python
for x1, x2, ... in dset:
    ...
```

The iterator. Recommend to use it inside the context. The unpacked variables `#!py x1, x2 ...` are ordered according to the given argument `#!py keywords` during the initialization.

-----

### :codicons-symbol-property: `size`

```python
dset.size
```

The size of the dataset. It contains the total number of samples for each epoch.

-----

### :codicons-symbol-property: `preproc`

```python
dset.preproc
```

The argument `#!py preprocfunc` during the initialziation. This property helps users to invoke the preprocessor manually.

## Example

???+ example "Example 1"
    === "Codes"
        ```python linenums="1"
        import numpy as np
        import mdnc

        class ProcCustom:
            def __init__(self, seed=1000, batch_size=16):
                self.batch_size = batch_size
                self.random_rng = np.random.default_rng(seed)

            def __call__(self, ds_x1, ds_x2):
                ind_x1 = np.sort(self.random_rng.integers(len(ds_x1), size=batch_size))
                ind_x2 = np.sort(self.random_rng.integers(len(ds_x2), size=batch_size))
                return ds_x1[ind_x1, ...], ds_x2[ind_x2, ...]

        dset = mdnc.data.h5py.H5RParser('test_rparser', keywords=['one', 'zero'], preprocfunc=ProcCustom())
        with dset.start() as p:
            for i, data in enumerate(p):
                print('data.h5py: Epoch 1, Batch {0}'.format(i), data[0].shape, data[1].shape)
        ```

???+ example "Example 2"
    === "Codes"
        ```python linenums="1"
        import numpy as np
        import mdnc

        class ProcCustom(mdnc.data.preprocs.ProcAbstract):
            def __init__(self, seed=1000, batch_size=16, inds=None, parent=None):
                super().__init__(inds=inds, parent=parent)
                self.batch_size = batch_size
                self.random_rng = np.random.default_rng(seed)

            def preprocess(self, ds):
                ind = np.sort(self.random_rng.integers(len(ds), size=batch_size))
                return ds[ind, ...]

            def postprocess(self, x):
                return x

        dset = mdnc.data.h5py.H5RParser('test_rparser', keywords=['one', 'zero'],
                                        preprocfunc=mdnc.data.preprocs.ProcScaler(parent=ProcCustom()))
        with dset.start() as p:
            for i, data in enumerate(p):
                print('data.h5py: Epoch 1, Batch {0}'.format(i), data[0].shape, data[1].shape)
        ```

[pydoc-pickable]:https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled "What can be pickled and unpickled?"
