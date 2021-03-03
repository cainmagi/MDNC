# data.h5py.H5SupSaver

:codicons-symbol-class: Class · [:octicons-file-code-24: Source]({{ source.root }}/data/h5py.py#L676)

```python
saver = mdnc.data.h5py.H5SupSaver(
    file_name=None, enable_read=False
)
```

Save supervised data set as `.h5` file. This class allows users to dump multiple datasets into one file handle, then it would save it as a `.h5` file. The keywords of the sets should be assigned by users. It supports both the context management and the dictionary-style nesting. It is built on top of `#!py h5py.Group` and `#!py h5py.Dataset`.

The motivation of using this saver includes:

* Provide an easier way for saving resizable datasets. All datasets created by this saver are resizable.
* Provide convenient APIs for creating `#!py h5py.Softlink`, `#!py h5py.Attributes` and `#!py h5py.VirtualDataSet`.
* Add context nesting supports for `#!py h5py.Group`. This would makes the codes more elegant.

## Arguments

**Requries**

| Argument {: .w-6rem} | Type {: .w-4rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `file_name` | `#!py str` | A path where we save the file. If not set, the saver would not open a file. |
| `enable_read` | `#!py bool` | When setting `#!py True`, enable the `a` mode. Otherwise, use `w` mode. This option is used when adding data to an existed file. |

## Methods

### :codicons-symbol-method: `config`

```python
saver.config(logver=0, **kwargs)
```

Make configuration for the saver. Only the explicitly given argument would be used for changing the configuration of this instance.

**Requries**

| Argument {: .w-5rem} | Type {: .w-4rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `logver` | `#!py int` | The verbose level of the outputs. When setting 0, would run silently. |
| `**kwargs` | | Any argument that would be used for creating `#!py h5py.Dataset`. The given argument would override the default value during the dataset creation. |

-----

### :codicons-symbol-method: `get_config`

```python
cfg = saver.get_config(name=None)
```

Get the current configuration value by the given `name`.

**Requries**

| Argument {: .w-5rem} | Type {: .w-4rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `name` | `#!py str` | The name of the required config value. |

**Returns**

| Argument {: .w-5rem} | Description {: .w-8rem} |
| :------: | :---------- |
| `cfg` | The required config value. |

-----

### :codicons-symbol-method: `open`

```python
saver.open(file_name, enable_read=None)
```

Open a new file. If a file has been opened before, this file would be closed. This method and the `__init__` method (need to specify `file_name`) support context management.

**Requries**

| Argument {: .w-6rem} | Type {: .w-4rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `file_name` | `#!py str` | A path where we save the file. |
| `enable_read` | `#!py bool` | When setting `#!py True`, enable the `a` mode. Otherwise, use `w` mode. This option is used when adding data to an existed file. If not set, the `enable_read` would be inherited from the class definition. Otherwise, the class definition `enable_read` would be updated by this new value. |

-----

### :codicons-symbol-method: `close`

```python
saver.close()
```

Close the saver.

-----

### :codicons-symbol-method: `dump`

```python
saver.dump(keyword, data, **kwargs)
```

Dump the dataset with a keyword into the file. The dataset is resizable, so this method could be used repeatly. The data would be always attached at the end of the current dataset.

**Requries**

| Argument {: .w-6rem} | Type {: .w-6rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `file_name` | `#!py str` | The keyword of the dumped dataset. |
| `data` | `#!py np.ndarray` | A new batch of data items, should be a numpy array. The axes `#!py data[1:]` should match the shape of existing dataset. |
| `**kwargs` | | Any argument that would be used for creating `#!py h5py.Dataset`. The given argument would override the default value and configs set by `#!py config()` during the dataset creation. |

-----

### :codicons-symbol-method: `set_link`

```python
saver.set_link(keyword, target, overwrite=True)
```

Create a h5py.Softlink.

**Requries**

| Argument {: .w-6rem} | Type {: .w-4rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `keyword` | `#!py str` | The keyword of the to-be created soft link. |
| `target` | `#!py str` | The reference (pointting position) of the soft link. |
| `overwrite` | `#!py bool` | if not `#!py True`, would skip this step when the the `keyword` exists. Otherwise, the `keyword` would be overwritten, even if it contains an `#!py h5py.Dataset`. |

-----

### :codicons-symbol-method: `set_attrs`

```python
saver.set_attrs(keyword, attrs=None, **kwargs)
```

Set attrs for an existed data group or dataset.

**Requries**

| Argument {: .w-6rem} | Type {: .w-4rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `keyword` | `#!py str` | The keyword where we set the attributes. |
| `attrs` | `#!py dict` | The attributes those would be set. |
| `**kwargs` | | More attributes those would be combined with `attrs` by `#!py dict.update()`. |

-----

### :codicons-symbol-method: `set_virtual_set`

```python
saver.set_virtual_set(keyword, sub_set_keys, fill_value=0.0)
```

Create a virtual dataset based on a list of subsets. All subsets require to be h5py.Dataset and need to share the same shape (excepting the first dimension, i.e. the sample number). The subsets would be concatenated at the `#!py axis=1`. For example, when `#!py d1.shape=[100, 20]`, `#!py d2.shape=[80, 20]`, the output virtual set would be `#!py d.shape=[100, 2, 20]`. In this case, `#!py d[80:, 1, :]` are filled by `fill_value`.

**Requries**

| Argument {: .w-7rem} | Type {: .w-5rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `keyword` | `#!py str` | The keyword of the dumped dataset. |
| `sub_set_keys` | `#!py (str, )` | A sequence of sub-set keywords. Each sub-set should share the same shape (except for the first dimension). |
| `fill_value` | `#!py float` | The value used for filling the blank area in the virtual dataset. |

## Properties

### :codicons-symbol-property: `attrs`

```python
attrs = saver.attrs  # Return the h5py.AttributeManager
saver.attrs = dict(...)  # Use a dictionary to update attrs.
```

Supports using a dictionary to update the attributes of the current `h5py` object. The returned `attrs` is used as `#!py h5py.AttributeManager`.

## Example

???+ example "Example 1"
    === "Codes"
        ```python linenums="1"
        import numpy as np
        import mdnc

        with mdnc.data.h5py.H5SupSaver('test_h5supsaver.h5', enable_read=False) as s:
            s.config(logver=1, shuffle=True, fletcher32=True, compression='gzip')
            s.dump('one', np.ones([25, 20]), chunks=(1, 20))
            s.dump('zero', np.zeros([25, 10]), chunks=(1, 10))
        ```

???+ example "Example 2"
    === "Codes"
        ```python linenums="1"
        import numpy as np
        import mdnc

        saver = mdnc.data.h5py.H5SupSaver(enable_read=False)
        saver.config(logver=1, shuffle=True, fletcher32=True, compression='gzip')
        with saver.open('test_h5supsaver.h5') as s:
            s.dump('test1', np.zeros([100, 20]))
            gb = s['group1']
            with gb['group2'] as g:
                g.dump('test2', np.zeros([100, 20]))
                g.dump('test2', np.ones([100, 20]))
                g.attrs = {'new': 1}
                g.set_link('test3', '/test1')
            print('data.h5py: Check open: s["group1"]={0}, s["group1/group2"]={1}'.format(gb.is_open, g.is_open))
        ```
