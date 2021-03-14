# data.preprocs.ProcMerge

:codicons-symbol-class: Class · [:octicons-file-code-24: Source]({{ source.root }}/data/preprocs.py#L262){ target="_blank" }

```python
proc = mdnc.data.preprocs.ProcMerge(
    procs=None, num_procs=None, parent=None
)
```

Merge manager. This processor is inhomogeneous, and designed for merging different processors by a more efficient way. For example,

```python
p = ProcMerge([Proc1(...), Proc2(...)])
```

Would apply `Proc1` to the first argument, and `Proc2` to the second argument. It is equivalent to

```python
p = Proc1(..., inds=0, parent=Proc2(..., inds=1))
```

This class should not be used if any sub-processor **does not** return the results with the same number of the input variables (out-arg changed). One exception is, the `parent` of this class could be an out-arg changed processor.

This API is more intuitive for users to concatenate serveral processors together. It will make your codes more readable and reduce the stack level of the processors.

## Arguments

**Requries**

| Argument {: .w-5rem} | Type {: .w-8rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `procs` | `#!py (ProcAbstract, )` | A sequence of processors. Each processor is derived from [`mdnc.data.preprocs.ProcAbstract`](../ProcAbstract). Could be used for initializing this merge processor. |
| `num_procs` | `#!py object` | The number of input arguments of this processor. If not set, would infer the number from the length of the argument `procs`. At least one of `procs` or `num_procs` needs to be specified. The two arguments could be specified together. |
| `parent` | [`ProcAbstract`](../ProcAbstract) | An instance derived from [`mdnc.data.preprocs.ProcAbstract`](../ProcAbstract). This instance would be used as the parent of the current instance. |

??? warning
    The argument `num_procs` should be greater than `procs`, if both `num_procs` and `procs` are specified.

## Methods

### :codicons-symbol-method: `preprocess`

```python
y_1, y_2, ... = proc.preprocess(x_1, x_2, ...)
```

The preprocess function. The n^th^ variable would be sent to the n^th^ processor configured for `proc`.

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

The postprocess function. The n^th^ variable would be sent to the n^th^ processor configured for `proc`.

If `parent` exists, the output of this function would be passed as the input of `#!py parent.postprocess()`. Otherwise, the output would be returned to users directly.

**Requries**

| Argument {: .w-5rem} | Type {: .w-6rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `(y, )` | `#!py np.ndarray` | A sequence of variables. Each variable comes from the next processors's outputs (if parent exists). The output of this method would be passed as the input of the parent's method. |

**Returns**

| Argument {: .w-5rem} | Description {: .w-8rem} |
| :------: | :---------- |
| `(x, )` | A sequence of `#!py np.ndarray`, the final postprocessed data. |

## Operators

### :codicons-symbol-operator: `#!py __getitem__`

```python
proc_i = proc[idx]
```

Get the i^th^ sub-processor.

??? warning
    If one sub-processor is managing multiple indicies, the returned sub-processor would always be same for those indicies. For example,

    === "Codes"
        ```python
        proc_m = Proc2(...)
        proc = ProcMerge([Proc1(...), proc_m, proc_m])
        proc_1 = proc[1]
        proc_2 = proc[2]
        print(proc_m is proc_1, proc_m is proc_2)
        ```

    This behavior is important if your `proc_m` is an inhomogeneous processor. It means although you get `proc_2` by `#!py proc[2]`, you still need to place your argument as the 2^nd^ input when using `proc_2`.

**Requries**

| Argument {: .w-5rem} | Type {: .w-4rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `idx` | `#!py int` | The index of the sub-processor. |

**Returns**

| Argument {: .w-5rem} | Description {: .w-8rem} |
| :------: | :---------- |
| `proc_i` | An instance derived from [`ProcAbstract`](../ProcAbstract), the i^th^ sub-processor. |

-----

### :codicons-symbol-operator: `#!py __setitem__`

```python
proc[idx] = proc_i
```

??? info
    This method supports multiple assignment, for example:

    === "Codes"
        ```python
        proc = ProcMerge(num_procs=3)
        proc[:] = Proc1(...)
        proc[1:2] = Proc2(...)
        ```

    This would be equivalent to

    === "Codes"
        ```python
        proc_m = Proc2(...)
        proc = ProcMerge([Proc1(...), proc_m, proc_m])
        ```

**Requries**

| Argument {: .w-5rem} | Type {: .w-6rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `idx` | `#!py int` or<br>`#!py slice` or<br>`#!py tuple` | The indicies that would be overwritten by the argument `proc_i`. |
| `proc_i` | [`ProcAbstract`](../ProcAbstract) | An instance derived from [`ProcAbstract`](../ProcAbstract), this sub-processor would be used for overriding one or more indicies. |

## Properties

### :codicons-symbol-variable: `num_procs`

```python
proc.num_procs
```

The number of sub-processors for this class. If one sub-processor is used for managing multiple indicies, it will be count for mutiple times.

-----

### :codicons-symbol-variable: `parent`

```python
proc.parent
```

The parent processor of this instance. The processor is also a derived class of [`ProcAbstract`](../ProcAbstract). If the parent does not exist, would return `#!py None`.

-----

### :codicons-symbol-variable: `has_ind`

```python
proc.has_ind
```

A bool flag, showing whether this processor and its all parent processors have `inds` configured. In this case, the arguments of [`preprocess()`](#preprocess) and [`postprocess()`](#postprocess) would not share the same operation. We call such kind of processors "Inhomogeneous processors".

Certianly, it will always be `#!py proc.has_ind=True` for this class.

## Examples

There are many kinds of method for using this class. For example,

???+ example "Example 1"
    === "Codes"
        ```python linenums="1"
        import numpy as np
        import mdnc

        proc = mdnc.data.preprocs.ProcMerge([mdnc.data.preprocs.ProcScaler(), mdnc.data.preprocs.ProcNSTScaler(dim=1)])
        random_rng = np.random.default_rng()
        x, y = random_rng.normal(loc=-1.0, scale=0.1, size=[5, 3]), random_rng.normal(loc=1.0, scale=3.0, size=[4, 2])
        x_, y_ = proc.preprocess(x, y)
        xr, yr = proc.postprocess(x_, y_)
        print('Processed shape:', x_.shape, y_.shape)
        print('Processed mean:', np.mean(x_), np.mean(y_))
        print('Processed range:', np.amax(np.abs(x_)), np.amax(np.abs(y_)))
        print('Inverse error:', np.amax(np.abs(x - xr)), np.amax(np.abs(y - yr)))
        ```

    === "Output"
        ```
        Processed shape: (5, 3) (4, 2)
        Processed mean: 4.440892098500626e-16 2.7755575615628914e-17
        Processed range: 1.0 1.0
        Inverse error: 0.0 0.0
        ```

???+ example "Example 2"
    === "Codes"
        ```python linenums="1"
        import numpy as np
        import mdnc

        proc1 = mdnc.data.preprocs.ProcScaler()
        proc2 = mdnc.data.preprocs.ProcNSTScaler(dim=1, inds=0, parent=mdnc.data.preprocs.ProcScaler(inds=1))
        proc = mdnc.data.preprocs.ProcMerge(num_procs=3)
        proc[0] = proc1
        proc[1:] = proc2
        random_rng = np.random.default_rng()
        x, y, z = random_rng.normal(loc=-1.0, scale=0.1, size=[5, 3]), random_rng.normal(loc=1.0, scale=3.0, size=[4, 2]), random_rng.normal(loc=1.0, scale=3.0, size=[4, 2])
        x_, y_, z_ = proc.preprocess(x, y, z)
        xr, yr, zr = proc.postprocess(x_, y_, z_)
        print('Processed shape:', x_.shape, y_.shape, z_.shape)
        print('Processed mean:', np.mean(x_), np.mean(y_), np.mean(z_))
        print('Processed range:', np.amax(np.abs(x_)), np.amax(np.abs(y_)), np.amax(np.abs(z_)))
        print('Inverse error:', np.amax(np.abs(x - xr)), np.amax(np.abs(y - yr)), np.amax(np.abs(z - zr)))
        ```

    === "Output"
        ```
        Processed shape: (5, 3) (4, 2) (4, 2)
        Processed mean: -1.7763568394002506e-16 -1.8041124150158794e-16 -1.314226505400029e-14
        Processed range: 1.0 1.0 1.0
        Inverse error: 0.0 1.1102230246251565e-16 0.0
        ```

This class could be also used for merge customized processor. But the customized processor should ensure the input and output numbers are the same, for example,

???+ example "Example 3"
    === "Codes"
        ```python linenums="1"
        import numpy as np
        import mdnc

        class ProcDerived(mdnc.data.preprocs.ProcAbstract):
            def __init__(self, a, parent=None):
                super().__init__(parent=parent, _disable_inds=True)
                self.a = a

            def preprocess(self, x, y):
                return self.a * x, (2 * self.a) * y

            def postprocess(self, x, y):
                return x / self.a, y / (2 * self.a)

        proc1 = mdnc.data.preprocs.ProcScaler()
        proc2 = mdnc.data.preprocs.ProcNSTScaler(dim=1, parent=ProcDerived(a=2.0))
        proc = mdnc.data.preprocs.ProcMerge(num_procs=3)
        proc[0] = proc1
        proc[1:] = proc2
        random_rng = np.random.default_rng()
        x, y, z = random_rng.normal(loc=-1.0, scale=0.1, size=[5, 3]), random_rng.normal(loc=1.0, scale=3.0, size=[4, 2]), random_rng.normal(loc=1.0, scale=3.0, size=[4, 2])
        x_, y_, z_ = proc.preprocess(x, y, z)
        xr, yr, zr = proc.postprocess(x_, y_, z_)
        print('Processed shape:', x_.shape, y_.shape, z_.shape)
        print('Processed mean:', np.mean(x_), np.mean(y_), np.mean(z_))
        print('Processed range:', np.amax(np.abs(x_)), np.amax(np.abs(y_)), np.amax(np.abs(z_)))
        print('Inverse error:', np.amax(np.abs(x - xr)), np.amax(np.abs(y - yr)), np.amax(np.abs(z - zr)))
        ```

    === "Output"
        ```
        Processed shape: (5, 3) (4, 2) (4, 2)
        Processed mean: -1.7763568394002506e-16 0.0 -5.273559366969494e-16
        Processed range: 1.0 1.0 1.0
        Inverse error: 0.0 2.220446049250313e-16 0.0
        ```
