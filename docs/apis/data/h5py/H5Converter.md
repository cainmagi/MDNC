# data.h5py.H5Converter

:codicons-symbol-class: Class · [:octicons-file-code-24: Source]({{ source.root }}/data/h5py.py#L82)

```python
converter = mdnc.data.h5py.H5Converter(
    file_name, oformat, to_other=True
)
```

Conversion between HDF5 data and other formats. The "other formats" would be arranged in to form of several nested folders and files. Each data group would be mapped into a folder, and each dataset would be mapped into a file.

???+ warning
    When the argument `to_other` is `#!py True`, the data would be converted to other formats. During this process, attributes would be lost, and the links and virtual datasets would be treated as `#!py h5py.Datasets`.

## Arguments

**Requries**

| Argument {: .w-5rem} | Type {: .w-4rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `file_name` | `#!py str` | A path where we find the dataset. If the conversion is from h5 to other, the path should refer a folder containing several subfiles, otherwise, it should refer an HDF5 file. |
| `oformat` | `#!py object` | The format function for a single dataset, it could be provided by users, or use the default configurations (`#!py str`). (avaliable: `#!py 'txt'`, `#!py 'bin'`.) |
| `to_other` | `#!py bool` | The flag for conversion mode. If set True, the mode would be h52other, i.e. an HDF5 set would be converted into other formats. If set False, the conversion would be reversed. |

??? tip
    The argument `oformat` could be a user defined custome object. It should provide two methods: `read()` and `write()`. An example of `txt` IO is shown as below:
    ```python linenums="1"
    import os
    import io
    import numpy as np

    class H52TXT:
        '''An example of converter between HDF5 and TXT'''
        def read(self, file_name):
            '''read function, for converting TXT to HDF5.
            file_name is the name of the single input file
            return an numpy array.'''
            with open(os.path.splitext(file_name)[0] + '.txt', 'r') as f:
                sizeText = io.StringIO(f.readline())
                sze = np.loadtxt(sizeText, dtype=np.int)
                data = np.loadtxt(f, dtype=np.float32)
                return np.reshape(data, sze)

        def write(self, h5data, file_name):
            '''write function, for converting HDF5 to TXT.
            h5data is the h5py.Dataset
            file_name is the name of the single output file.
            '''
            with open(os.path.splitext(file_name)[0] + '.txt', 'w') as f:
                np.savetxt(f, np.reshape(h5data.shape, (1, h5data.ndim)), fmt='%d')
                if h5data.ndim > 1:
                    for i in range(h5data.shape[0]):
                        np.savetxt(f, h5data[i, ...].ravel(), delimiter='\n')
                else:
                    np.savetxt(f, h5data[:].ravel(), delimiter='\n')

    converter = mdnc.data.h5py.H5Converter(
        ..., oformat=H52TXT()
    )
    ```

## Methods

### :codicons-symbol-method: `convert`

```python
converter.convert()
```

Perform the data conversion.

## Examples

???+ example
    === "Codes"
        ```python linenums="1"
        import mdnc

        cvt_o = mdnc.data.h5py.H5Converter('test_converter.h5', 'txt', to_other=True)
        cvt_o.convert()  # From HDF5 dataset to txt files.
        cvt_i = mdnc.data.h5py.H5Converter('test_converter.h5', 'txt', to_other=False)
        cvt_i.convert()  # From txt files to HDF5 dataset.
        ```
