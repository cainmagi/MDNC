# data.preprocs.ProcScaler

:codicons-symbol-class: Class · [:octicons-file-code-24: Source]({{ source.root }}/data/h5py.py#L1351)

```python
proc = mdnc.data.preprocs.ProcScaler(
    shift=None, scale=None, axis=-1, inds=None, parent=None
)
```

This is a homogeneous processor. It accepts two variables `shift` ($\mu$), `scale` ($\sigma$) to perform the following normalization:

```math
\begin{align}
    \mathbf{y}_n = \frac{1}{\sigma} ( \mathbf{x}_n - \mu ),
\end{align}
```

where $\mathbf{x}_n$ and $\mathbf{y}_n$ are the i^th^ input argument and the corresponding output argument respectively.

* If not setting $\mu$, would use the mean value of the input mini-batch to shift the argument, i.e. $\mu_n = \bar{\mathbf{x}_n}$;
* If not setting $\sigma$, would use the max-abs value of the input mini-batch to scale the argument, i.e. $\sigma_n = \max |\mathbf{x}_n - \mu_n|$.

The above two caulation is estimated on mini-batches. This configuration may cause unstable issues when the input mini-batches are not i.i.d.. Therefore, we recommend users to always set `shift` and `scale` manually.

## Arguments

**Requries**

| Argument {: .w-5rem} | Type {: .w-6rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `shift` | `#!py int` or<br>`#!py np.ndarray` | The $\mu$ variable used for shifting the mean value of mini-batches. This value could be an `#!py np.ndarray` supporting broadcasting. If set `#!py None`, would shift the mean value of each mini-batch to 0. |
| `scale` | `#!py int` or<br>`#!py np.ndarray` | The $\sigma$ variable used for shifting the mean value of mini-batches. This value could be an `#!py np.ndarray` supporting broadcasting. If set `#!py None`, would scale the max-abs value of each mini-batch to 1. |
| `axis` | `#!py int` or<br>`#!py (int, )` | The axis used for calculating the normalization parameters. If given a sequence, would calculate the paramters among higher-dimensional data. Only used when `shift` or `scale` is not `#!py None`. |
| `inds` | `#!py int` or<br>`#!py (int, )` | Index or indicies of variables where the user implemented methods would be broadcasted. The variables not listed in this argument would be passed to the output without any processing. If set `#!py None`, methods would be broadcasted to all variables. |
| `parent` | [`ProcAbstract`](../ProcAbstract) | Another instance derived from [`mdnc.data.preprocs.ProcAbstract`](../ProcAbstract). The output of `#!py parent.preprocess()` would be used as the input of `#!py self.preprocess()`. The input of `#!py self.postprocess()` would be used as the input of `#!py parent.preprocess()`. |

## Methods

### :codicons-symbol-method: `preprocess`

```python
y_1, y_2, ... = proc.preprocess(x_1, x_2, ...)
```

The preprocess function. Calculate the re-scaled values from the input variables.

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

The postprocess function. The inverse operator of the scaling.

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

### :codicons-symbol-variable: `shift`

```python
proc.shift
```

The shifting value $\mu$.

-----

### :codicons-symbol-variable: `scale`

```python
proc.scale
```

The scaling value $\sigma$.

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
        import mdnc

        proc = mdnc.data.preprocs.ProcScaler(shift=1.0, scale=3.0)
        random_rng = np.random.default_rng()
        x, y = random_rng.normal(loc=1.0, scale=3.0, size=[5, 4]), random_rng.normal(loc=1.0, scale=6.0, size=[7, 5])
        x_, y_ = proc.preprocess(x, y)
        xr, yr = proc.postprocess(x_, y_)
        print('Processed shape:', x_.shape, y_.shape)
        print('Processed mean:', np.mean(x_), np.mean(y_))
        print('Processed std:', np.std(x_), np.std(y_))
        print('Inverse error:', np.amax(np.abs(x - xr)), np.amax(np.abs(y - yr)))
        ```

???+ example "Example 2"
    === "Codes"
        ```python linenums="1"
        import numpy as np
        from sklearn import preprocessing
        import mdnc

        random_rng = np.random.default_rng()
        x = random_rng.normal(loc=1.0, scale=3.0, size=[100, 5])
        rsc = preprocessing.RobustScaler()
        rsc.fit(x)
        proc = mdnc.data.preprocs.ProcScaler(shift=np.expand_dims(rsc.center_, axis=0), scale=np.expand_dims(rsc.scale_, axis=0))
        x_ = proc.preprocess(x)
        x_sl = rsc.transform(x)
        x_r = proc.postprocess(x_)
        x_r_sl = rsc.inverse_transform(x_sl)
        print('Processed error:', np.amax(np.abs(x_ - x_sl)))
        print('Inverse error:', np.amax(np.abs(x_r - x_r_sl)))
        ```
