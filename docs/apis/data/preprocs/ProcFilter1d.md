# data.preprocs.ProcFilter1d

:codicons-symbol-class: Class · [:octicons-file-code-24: Source]({{ source.root }}/data/h5py.py#L1351)

```python
proc = mdnc.data.preprocs.ProcFilter1d(
    axis=-1, band_low=3.0, band_high=15.0, nyquist=500.0,
    filter_type='butter', out_type='sosfilt2', filter_args=None,
    inds=None, parent=None
)
```

This is a homogeneous processor. It is an implementation of the 1D IIR band-pass filters (also supports low-pass or high-pass filter).

The IIR filer would be only performed on the chosen axis. If users want to filter the data along multiple dimensions, using a stack of this instance may be needed, for example:

```python
proc = ProcFilter1d(axis=1, parent=ProcFilter1d(axis=2, ...))
```

???+ warning
    Plese pay attention to the results. This operation is not invertible, and the [`postprocess()`](#postprocess) would do nothing.

## Arguments

**Requries**

| Argument {: .w-6rem} | Type {: .w-6rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `axis`      | `#!py int` | The axis where we apply the 1D filter. |
| `band_low`  | `#!py float` | The lower cut-off frequency. If only set this value, the filter would become high-pass. |
| `band_high` | `#!py float` | The higher cut-off frequency. If only set this value, the filter become high-pass. |
| `nyquist`   | `#!py float` | The nyquist frequency of the data. |
| `filter_type` | `#!py str` | The IIR filter type, could be: `#!py 'butter'`, `#!py 'cheby1'`, `#!py 'cheby2'`, `#!py 'ellip'`, or `#!py 'bessel'`. See the filter type list to check the details. |
| `out_type`    | `#!py str` | The output filter paramter type, could be `#!py 'sosfilt2'`, `#!py 'filt2'`, `#!py 'sos'`, `#!py 'ba'`. See the out type list to check the details. |
| `filter_args` | `#!py str` | A dictionary including other filter arguments, not all arguments are required for each filter, could contain `#!py 'order'`, `#!py 'ripple'`, `#!py 'attenuation'`. See the filter argument list to check the details. |
| `inds` | `#!py int` or<br>`#!py (int, )` | Index or indicies of variables where the user implemented methods would be broadcasted. The variables not listed in this argument would be passed to the output without any processing. If set `#!py None`, methods would be broadcasted to all variables. |
| `parent` | [`ProcAbstract`](../ProcAbstract) | Another instance derived from [`mdnc.data.preprocs.ProcAbstract`](../ProcAbstract). The output of `#!py parent.preprocess()` would be used as the input of `#!py self.preprocess()`. The input of `#!py self.postprocess()` would be used as the input of `#!py parent.preprocess()`. |

**Filter types**

| Argument {: .w-5rem} | Description {: .w-8rem} |
| :------: | :---------- |
| `butter` | Butterworth IIR filter, see [:fontawesome-solid-external-link-alt: scipy.signal.butter][scipy-butter]. |
| `cheby1` | Chebyshev type I IIR filter, see [:fontawesome-solid-external-link-alt: scipy.signal.cheby1][scipy-cheby1]. |
| `cheby2` | Chebyshev type II IIR filter, see [:fontawesome-solid-external-link-alt: scipy.signal.cheby2][scipy-cheby2]. |
| `ellip`  | Elliptic (Cauer) IIR filter, see [:fontawesome-solid-external-link-alt: scipy.signal.ellip][scipy-ellip]. |
| `bessel` | Bessel/Thomson IIR filter, see [:fontawesome-solid-external-link-alt: scipy.signal.bessel][scipy-bessel]. |

**Out types**

| Argument {: .w-5rem} | Description {: .w-8rem} |
| :------: | :---------- |
| `sosfilt2` | Forward-backward second-order filter coefficients, see [:fontawesome-solid-external-link-alt: scipy.signal.sosfiltfilt][scipy-sosfiltfilt]. |
| `filt2` | Forward-backward first-order filter coefficients, see [:fontawesome-solid-external-link-alt: scipy.signal.filtfilt][scipy-filtfilt]. |
| `sos` | Second-order filter coefficients, see [:fontawesome-solid-external-link-alt: scipy.signal.sosfilt][scipy-sosfilt]. |
| `ba`  | First-order filter coefficients, see [:fontawesome-solid-external-link-alt: scipy.signal.lfilter][scipy-lfilter]. |

**Filter arguments**

The arguments in the following table are the default values of the `filter_args`. If one value is marked as :fontawesome-solid-times:, it means the argument is not available for this filter.

| Argument {: .w-5rem} | `butter` {: .w-4rem} | `cheby1` {: .w-4rem} | `cheby2` {: .w-4rem} | `ellip` {: .w-4rem} | `bessel` {: .w-4rem} |
| :------: | :-----: | :-----: | :-----: | :-----: | :-----: |
| `order`       | `#!py 10`                 | `#!py 4`                  | `#!py 10`                 | `#!py 4`                  | `#!py 10`                 |
| `ripple`      | :fontawesome-solid-times: | `#!py 5`                  | :fontawesome-solid-times: | `#!py 5`                  | :fontawesome-solid-times: |
| `attenuation` | :fontawesome-solid-times: | :fontawesome-solid-times: | `#!py 40`                 | `#!py 40`                 | :fontawesome-solid-times: |

## Methods

### :codicons-symbol-method: `preprocess`

```python
y_1, y_2, ... = proc.preprocess(x_1, x_2, ...)
```

The preprocess function. Calculate the filterd results for each argument.

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

The postprocess function. Nothing would be done during the post-processing stage of this processor, i.e. `#!py x = proc.postprocess(x)`.

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

        proc = mdnc.data.preprocs.ProcFilter1d(axis=-1, filter_type='butter', band_low=3.0, band_high=15.0, nyquist=100)
        random_rng = np.random.default_rng()
        data = random_rng.uniform(low=1-0.01, high=1+0.01, size=[1, 1024])
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

[scipy-butter]:https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html "scipy.signal.butter"
[scipy-cheby1]:https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheby1.html "scipy.signal.cheby1"
[scipy-cheby2]:https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheby2.html "scipy.signal.cheby2"
[scipy-ellip]:https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ellip.html "scipy.signal.ellip"
[scipy-bessel]:https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.bessel.html "scipy.signal.bessel"
[scipy-sosfiltfilt]:https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html "scipy.signal.sosfiltfilt"
[scipy-filtfilt]:https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html "scipy.signal.filtfilt"
[scipy-sosfilt]:https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html "scipy.signal.sosfilt"
[scipy-lfilter]:https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html "scipy.signal.lfilter"
