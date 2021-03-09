# data.preprocs.ProcLifter

:codicons-symbol-class: Class · [:octicons-file-code-24: Source]({{ source.root }}/data/h5py.py#L1351)

```python
proc = mdnc.data.preprocs.ProcLifter(
    a, inds=None, parent=None
)
```

This is a homogeneous processor. It use the parameter `a` to perform such an invertible transform:

```math
\mathbf{y}_n = \mathrm{sign} (\mathbf{x}_n) * \log (1 + a * |\mathbf{x}_n|)
```

This transform could strengthen the low-amplitude parts of the signal, because the data is transformed into the log domain.

## Arguments

**Requries**

| Argument {: .w-5rem} | Type {: .w-6rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `a` | `#!py float` | The parameter used for log-lifting the data. |
| `inds` | `#!py int` or<br>`#!py (int, )` | Index or indicies of variables where the user implemented methods would be broadcasted. The variables not listed in this argument would be passed to the output without any processing. If set `#!py None`, methods would be broadcasted to all variables. |
| `parent` | [`ProcAbstract`](../ProcAbstract) | Another instance derived from [`mdnc.data.preprocs.ProcAbstract`](../ProcAbstract). The output of `#!py parent.preprocess()` would be used as the input of `#!py self.preprocess()`. The input of `#!py self.postprocess()` would be used as the input of `#!py parent.preprocess()`. |

## Methods

### :codicons-symbol-method: `preprocess`

```python
y_1, y_2, ... = proc.preprocess(x_1, x_2, ...)
```

The preprocess function. Perform the log-lifting.

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

The postprocess function. The inverse operator of the lifting.

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

### :codicons-symbol-variable: `a`

```python
proc.a
```

The lifting parameter $a$.

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

???+ example "Example"
    === "Codes"
        ```python linenums="1"
        import numpy as np
        import matplotlib.pyplot as plt
        import mdnc

        t = np.linspace(-2 * np.pi, 2 * np.pi, 200)
        proc = mdnc.data.preprocs.ProcLifter(a=10.0)
        x = np.cos(np.pi * t) + 0.5 * np.cos(1.5 * np.pi * t + 0.1) + 0.2 * np.cos(2.5 * np.pi * t + 0.3) + 0.1 * np.cos(3.5 * np.pi * t + 0.7)
        x_ = proc.preprocess(x)
        xr = proc.postprocess(x_)

        fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(12, 5))
        axs[0].plot(t, x_)
        axs[1].plot(t, xr)
        axs[2].plot(t, x)
        axs[0].set_ylabel('Preprocessing')
        axs[1].set_ylabel('Inversed preprocessing')
        axs[2].set_ylabel('Raw data')
        plt.tight_layout()
        plt.show()
        ```
