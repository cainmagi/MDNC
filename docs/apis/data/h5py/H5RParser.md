# data.h5py.H5RParser

:codicons-symbol-class: Class · :codicons-symbol-field: Context · [:octicons-file-code-24: Source]({{ source.root }}/data/h5py.py#L1472){ target="_blank" }

```python
dset = mdnc.data.h5py.H5RParser(
    file_name, keywords, preprocfunc, batch_num=100,
    num_workers=4, num_buffer=10
)
```

This class allows users to feed one .h5 file, and convert it to [`mdnc.data.sequence.MPSequence`](../../sequence/MPSequence). The realization could be described as:

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

Start the process pool. This method is implemented by [`mdnc.data.sequence.MPSequence`](../../sequence/MPSequence). It supports context management.

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
    Even if you set `#!py shuffle=False`, due to the mechanism of the parallelization, the sample order during the iteration may still get a little bit shuffled. To ensure your sample order not changed, please use `#!py shuffle=False` during the initialization and use [`start_test()`](#start_test) instead.

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
    This method also supports context management. See [`start()`](#start) to check how to use it.

-----

### :codicons-symbol-method: `finish`

```python
dset.finish()
```

Finish the process pool. The compatible mode would be auto detected by the previous `start()`.

## Properties

### :codicons-symbol-variable: `len()`, `batch_num`

```python
len(dset)
dset.batch_num
```

The length of the dataset. It is the number of mini-batches, also the number of iterations for each epoch.

-----

### :codicons-symbol-variable: `iter()`

```python
for x1, x2, ... in dset:
    ...
```

The iterator. Recommend to use it inside the context. The unpacked variables `#!py x1, x2 ...` are ordered according to the given argument `#!py keywords` during the initialization.

-----

### :codicons-symbol-variable: `size`

```python
dset.size
```

The size of the dataset. It contains the total number of samples for each epoch.

-----

### :codicons-symbol-variable: `preproc`

```python
dset.preproc
```

The argument `#!py preprocfunc` during the initialziation. This property helps users to invoke the preprocessor manually.

## Examples

???+ example "Example 1"
    === "Codes"
        ```python linenums="1"
        import os
        import numpy as np
        import mdnc

        root_folder = 'alpha-test'
        os.makedirs(root_folder, exist_ok=True)

        class ProcCustom:
            def __init__(self, seed=1000, batch_size=16):
                self.batch_size = batch_size
                self.random_rng = np.random.default_rng(seed)

            def __call__(self, ds_x1, ds_x2):
                ind_x1 = np.sort(self.random_rng.choice(len(ds_x1), replace=False, size=self.batch_size))
                ind_x2 = np.sort(self.random_rng.choice(len(ds_x2), replace=False, size=self.batch_size))
                return ds_x1[ind_x1, ...], ds_x2[ind_x2, ...]

        if __name__ == '__main__':
            # Prepare the datasets.
            set_list_file = os.path.join(root_folder, 'web-data')
            mdnc.data.webtools.DataChecker.init_set_list(set_list_file)
            dc = mdnc.data.webtools.DataChecker(root=root_folder, set_list_file=set_list_file, token='', verbose=False)
            dc.add_query_file('test_data_h5gparser.h5')
            dc.query()

            # Perform test.
            dset = mdnc.data.h5py.H5RParser(os.path.join(root_folder, 'test_data_h5gparser'),
                                            keywords=['one', 'zero'], preprocfunc=ProcCustom())
            with dset.start() as p:
                for i, data in enumerate(p):
                    print('data.h5py: Epoch 1, Batch {0}'.format(i), data[0].shape, data[1].shape)
        ```

    === "Output"
        ```
        data.webtools: All required datasets are available.
        data.h5py: Epoch 1, Batch 0 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 1 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 2 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 3 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 4 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 5 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 6 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 7 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 8 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 9 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 10 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 11 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 12 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 13 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 14 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 15 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 16 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 17 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 18 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 19 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 20 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 21 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 22 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 23 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 24 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 25 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 26 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 27 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 28 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 29 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 30 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 31 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 32 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 33 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 34 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 35 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 36 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 37 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 38 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 39 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 40 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 41 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 42 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 43 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 44 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 45 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 46 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 47 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 48 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 49 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 50 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 51 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 52 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 53 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 54 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 55 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 56 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 57 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 58 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 59 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 60 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 61 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 62 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 63 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 64 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 65 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 66 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 67 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 68 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 69 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 70 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 71 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 72 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 73 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 74 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 75 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 76 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 77 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 78 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 79 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 80 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 81 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 82 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 83 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 84 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 85 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 86 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 87 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 88 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 89 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 90 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 91 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 92 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 93 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 94 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 95 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 96 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 97 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 98 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 99 torch.Size([16, 20]) torch.Size([16, 10])
        ```

???+ example "Example 2"
    === "Codes"
        ```python linenums="1"
        import os
        import numpy as np
        import mdnc

        root_folder = 'alpha-test'
        os.makedirs(root_folder, exist_ok=True)

        class ProcCustom(mdnc.data.preprocs.ProcAbstract):
            def __init__(self, seed=1000, batch_size=16, inds=None, parent=None):
                super().__init__(inds=inds, parent=parent)
                self.batch_size = batch_size
                self.random_rng = np.random.default_rng(seed)

            def preprocess(self, ds):
                ind = np.sort(self.random_rng.choice(len(ds), replace=False, size=self.batch_size))
                return ds[ind, ...]

            def postprocess(self, x):
                return x

        if __name__ == '__main__':
            # Prepare the datasets.
            set_list_file = os.path.join(root_folder, 'web-data')
            mdnc.data.webtools.DataChecker.init_set_list(set_list_file)
            dc = mdnc.data.webtools.DataChecker(root=root_folder, set_list_file=set_list_file, token='', verbose=False)
            dc.add_query_file('test_data_h5gparser.h5')
            dc.query()

            # Perform test.
            dset = mdnc.data.h5py.H5RParser(os.path.join(root_folder, 'test_data_h5gparser'),
                                            keywords=['one', 'zero'], preprocfunc=ProcCustom())
            with dset.start() as p:
                for i, data in enumerate(p):
                    print('data.h5py: Epoch 1, Batch {0}'.format(i), data[0].shape, data[1].shape)
        ```

    === "Output"
        ```
        data.webtools: All required datasets are available.
        data.h5py: Epoch 1, Batch 0 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 1 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 2 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 3 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 4 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 5 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 6 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 7 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 8 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 9 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 10 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 11 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 12 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 13 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 14 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 15 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 16 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 17 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 18 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 19 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 20 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 21 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 22 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 23 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 24 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 25 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 26 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 27 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 28 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 29 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 30 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 31 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 32 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 33 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 34 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 35 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 36 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 37 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 38 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 39 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 40 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 41 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 42 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 43 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 44 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 45 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 46 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 47 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 48 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 49 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 50 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 51 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 52 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 53 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 54 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 55 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 56 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 57 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 58 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 59 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 60 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 61 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 62 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 63 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 64 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 65 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 66 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 67 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 68 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 69 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 70 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 71 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 72 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 73 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 74 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 75 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 76 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 77 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 78 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 79 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 80 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 81 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 82 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 83 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 84 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 85 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 86 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 87 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 88 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 89 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 90 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 91 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 92 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 93 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 94 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 95 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 96 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 97 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 98 torch.Size([16, 20]) torch.Size([16, 10])
        data.h5py: Epoch 1, Batch 99 torch.Size([16, 20]) torch.Size([16, 10])
        ```

[pydoc-picklable]:https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled "What can be pickled and unpickled?"
