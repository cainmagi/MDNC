# data.preprocs.ProcPad

:codicons-symbol-class: Class · [:octicons-file-code-24: Source]({{ source.root }}/data/h5py.py#L1351)

```python
proc = mdnc.data.preprocs.ProcPad(
    pad_width, inds=None, parent=None, **kwargs
)
```

This is a homogeneous processor. Use [`np.pad`][numpy-pad] to pad the data.

This processor supports all [`np.pad`][numpy-pad] options. Besides, this processor also support cropping. If any element in the argument `pad_width` is negative, would perform cropping on that axis. For example:

```python
p = ProcPad(pad_width=((5, -5),))
y = p(x)  # x.shape=(20,), y.shape=(20,)
```

In this case, the data is padded by 5 samples at the beginning, but cropped 5 samples at the end.

This operator is not invertible when cropping is applied. The postprocess would try to revert the padding / cropping configurations to match the input data.

???+ warning
    If cropping is used, this processor would be not invertible (unless we have the argument `#!py mode='wrap'`). The [`postprocess()`](#postprocess) method would try to pad the cropped part with the processed data.

## Arguments

**Requries**

| Argument {: .w-5rem} | Type {: .w-8rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `pad_width` | `#!py int` or<br>`#!py (int, int)`<br>`#!py ((int, int), ...)` | The `padding_width` argument of the `np.pad` function. If any element is negative, it means this elment is a cropping size. This argument only supports 3 cases: <ul> <li>`#!py width`: use the same padding / cropping width along all axes.</li> <li>`#!py (begin, end)`: use the same padding / cropping length for both edges along all axes.</li> <li>`#!py ((begin, end), ...)`: use different padding / cropping lengths for both edges along each axis.</li> </ul> |
| `inds` | `#!py int` or<br>`#!py (int, )` | Index or indicies of variables where the user implemented methods would be broadcasted. The variables not listed in this argument would be passed to the output without any processing. If set `#!py None`, methods would be broadcasted to all variables. |
| `parent` | [`ProcAbstract`](../ProcAbstract) | Another instance derived from [`mdnc.data.preprocs.ProcAbstract`](../ProcAbstract). The output of `#!py parent.preprocess()` would be used as the input of `#!py self.preprocess()`. The input of `#!py self.postprocess()` would be used as the input of `#!py parent.preprocess()`. |

## Methods

### :codicons-symbol-method: `preprocess`

```python
y_1, y_2, ... = proc.preprocess(x_1, x_2, ...)
```

The preprocess function. Perform the padding / cropping.

If `parent` exists, the input of this function comes from the output of `#!py parent.preprocess()`. Otherwise, the input would comes from the input varibable directly.

**Requries**

| Argument {: .w-5rem} | Type {: .w-6rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `(x, )` | `#!py np.ndarray` | A sequence of variables. Each variable comes from the parent's outputs (if parent exists). The output of this method would be passed as the input of the next processor (if this processor is used as parent). |

**Returns**

| Argument {: .w-5rem} | Description {: .w-8rem} |
| :------: | :---------- |
| `(y, )` | A sequence of `#!py np.ndarray`, the final preprocessed data. |

-----

### :codicons-symbol-method: `postprocess`

```python
x_1, x_2, ... = proc.postprocess(y_1, y_2, ...)
```

The postprocess function. The inverse operator is not invertible if cropping is used in [`preprocess()`](#preprocess). In this case, the cropped part would be padded by processed data `(y, )`.

If `parent` exists, the output of this function would be passed as the input of `#!py parent.postprocess()`. Otherwise, the output would be returned to users directly.

**Requries**

| Argument {: .w-5rem} | Type {: .w-6rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `(y, )` | `#!py np.ndarray` | A sequence of variables. Each variable comes from the next processors's outputs (if parent exists). The output of this method would be passed as the input of the parent's method. |

**Returns**

| Argument {: .w-5rem} | Description {: .w-8rem} |
| :------: | :---------- |
| `(x, )` | A sequence of `#!py np.ndarray`, the final postprocessed data. |

## Properties

### :codicons-symbol-variable: `pad_width`

```python
proc.pad_width
```

The padding width of the processor.

-----

### :codicons-symbol-variable: `parent`

```python
proc.parent
```

The parent processor of this instance. The processor is also a derived class of `ProcAbstract`. If the parent does not exist, would return `#!py None`.

-----

### :codicons-symbol-variable: `has_ind`

```python
proc.has_ind
```

A bool flag, showing whether this processor and its all parent processors have `inds` configured or initialized with `_disable_inds`. In this case, the arguments of [`preprocess()`](#preprocess) and [`postprocess()`](#postprocess) would not share the same operation. We call such kind of processors "Inhomogeneous processors".

## Examples

The processor need to be derived. We have two ways to implement the derivation, see the following examples.

???+ example "Example 1"
    === "Codes"
        ```python linenums="1"
        import numpy as np
        import matplotlib.pyplot as plt
        import mdnc

        proc = mdnc.data.preprocs.ProcPad(pad_width=((0, 0), (10, -10)), mode='wrap')
        random_rng = np.random.default_rng()
        data = random_rng.uniform(low=0.0, high=1.0, size=[10, 30])
        t = proc.preprocess(data)
        t_b = proc.postprocess(t)

        fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(12, 5))
        axs[0].plot(t[0])
        axs[1].plot(t_b[0])
        axs[2].plot(data[0])
        axs[0].set_ylabel('Preprocessing')
        axs[1].set_ylabel('Inversed preprocessing')
        axs[2].set_ylabel('Raw data')
        plt.tight_layout()
        plt.show()
        ```

???+ example "Example 2"
    === "Codes"
        ```python linenums="1"
        import numpy as np
        import matplotlib.pyplot as plt
        import mdnc

        proc = mdnc.data.preprocs.ProcPad(pad_width=((0, 0), (10, -10), (-10, 10)), mode='constant', constant_values=0.0)
        random_rng = np.random.default_rng()
        data = random_rng.uniform(low=0.0, high=1.0, size=[10, 30, 30])
        t = proc.preprocess(data)
        t_b = proc.postprocess(t)

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
        ```

[numpy-pad]:https://numpy.org/doc/stable/reference/generated/numpy.pad.html "numpy.pad"
