# data.preprocs.ProcAbstract

:codicons-symbol-class: Abstract Class · [:octicons-file-code-24: Source]({{ source.root }}/data/h5py.py#L1351)

```python
proc = mdnc.data.preprocs.ProcAbstract(
    inds=None, parent=None, _disable_inds=False
)
```

The basic processor class supporting cascading and variable-level broadcasting:

* Cascading: It means the derived class of this abstract class will support using such a method `#!py ProcDerived(parent=ProcDerived(...))` to create composition of processors.
* Variable level broadcasting: It means when `#!py _disable_inds=False` the user-implemented methods, for example, `#!py def preprocess(x)`, would be broadcasted to arbitrary number of input variables, like `proc.preprocess(x1, x2, ...)`.

???+ info
    This is an abstract class, which means you **could not** create an instance of this class by codes like this

    ```python
    proc = ProcAbstract(...)
    ```

    The correct way to use this class it to implement a derived class from this class. The intertage has 2 requirements:
    
    1. The `#!py __init__` method of **this** class need to be called **inside** the `#!py __init__` method of the **derived** class.
    2. The [`preprocess()`](#preprocess) and [`postprocess()`](#postprocess) methods need to be implemented.

    We recommend to expose the argument `inds` and `parent` in the derived class. But `_disable_inds` should not be accessed by users. See [Examples](#examples) to view how to make the derivation.

## Arguments

**Requries**

| Argument {: .w-7rem} | Type {: .w-6rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `inds` | `#!py int` or<br>`#!py (int, )` | Index or indicies of variables where the user implemented methods would be broadcasted. The variables not listed in this argument would be passed to the output without any processing. If set `#!py None`, methods would be broadcasted to all variables. |
| `parent` | `#!py ProcAbstract` | Another derived class of `ProcAbstract`. The output of `#!py parent.preprocess()` would be used as the input of `#!py self.preprocess()`. The input of `#!py self.postprocess()` would be used as the input of `#!py parent.preprocess()`. |
| `_disable_inds` | `#!py bool` | A flag used by developers. If set `#!py True`, the broadcasting would not be used. It means that the user-implemented arguments would be exactly the arguments to be used. |

??? warning
    The argument `inds` and `parent` in the derived class. But `_disable_inds` should not be accessed by users. See [Examples](#examples) to view how to make the derivation.

## Abstract Methods

### :codicons-symbol-method: `preprocess`

```python
y_1, y_2, ... = proc.preprocess(x_1, x_2, ...)
```

The preprocess function. If `parent` exists, the input of this function comes from the output of `#!py parent.preprocess()`. Otherwise, the input would comes from the input varibable directly.

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

The postprocess function. If `parent` exists, the output of this function would be passed as the input of `#!py parent.postprocess()`. Otherwise, the output would be returned to users directly.

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

???+ example "Example 1: with inds"
    === "Codes"
        ```python linenums="1"
        import numpy as np
        import mdnc

        class ProcDerived(mdnc.data.preprocs.ProcAbstract):
            def __init__(self, a, inds=None, parent=None):
                super().__init__(inds=inds, parent=parent)
                self.a = a

            def preprocess(self, x): # The input is an np.ndarray
                return self.a * x

            def postprocess(self, x): # The inverse operator
                return x / self.a

        proc = ProcDerived(a=2.0)
        x, y, z = np.ones([5, 2]), np.ones([3, 2]), np.ones([4, 3])
        x_, y_, z_ = proc.preprocess(x, y, z)
        xr, yr, zr = proc.postprocess(x_, y_, z_)
        print('Processed shape:', x_.shape, y_.shape, z_.shape)
        print('Processed error:', np.amax(np.abs(x_ - 2 * x)), np.amax(np.abs(y_ - 2 * y)), np.amax(np.abs(z_ - 2 * z)))
        print('Inverse error:', np.amax(np.abs(x - xr)), np.amax(np.abs(y - yr)), np.amax(np.abs(z - zr)))

        proc2 = ProcDerived(a=2.0, inds=[1, 2])
        x_, y_, z_ = proc.preprocess(x, y, z)
        xr, yr, zr = proc.postprocess(x_, y_, z_)
        print('Processed error:', np.amax(np.abs(x_ - x)), np.amax(np.abs(y_ - 2 * y)), np.amax(np.abs(z_ - 2 * z)))
        print('Inverse error:', np.amax(np.abs(x - xr)), np.amax(np.abs(y - yr)), np.amax(np.abs(z - zr)))
        ```

???+ example "Example 2: without inds"
    === "Codes"
        ```python linenums="1"
        import numpy as np
        import mdnc

        class ProcDerived(mdnc.data.preprocs.ProcAbstract):
            def __init__(self, a, parent=None):
                super().__init__(parent=parent, _disable_inds=True)
                self.a = a

            def preprocess(self, x, y, z): # All inputs are arrays
                return self.a * x, self.a * y, self.a * z

            def postprocess(self, x, y, z): # The inverse operator
                return x / self.a, y / self.a, z / self.a

        proc = ProcDerived(a=2.0)
        x, y, z = np.ones([5, 2]), np.ones([3, 2]), np.ones([4, 3])
        x_, y_, z_ = proc.preprocess(x, y, z)
        xr, yr, zr = proc.postprocess(x_, y_, z_)
        print('Processed shape:', x_.shape, y_.shape, z_.shape)
        print('Processed error:', np.amax(np.abs(x_ - 2 * x)), np.amax(np.abs(y_ - 2 * y)), np.amax(np.abs(z_ - 2 * z)))
        print('Inverse error:', np.amax(np.abs(x - xr)), np.amax(np.abs(y - yr)), np.amax(np.abs(z - zr)))
        ```

In the above two examples, the processor would multiply the inputs by `#!py 2.0`. The first implementation allows users to use the argument `inds` to determine which variables require to be processed. The user-implemented methods in the second example would fully control the input and output arguments.

Actually, the second implementation allows user to change the number of output arguments, for example:

???+ example "Example 3: out args changed"
    === "Codes"
        ```python linenums="1"
        import numpy as np
        import mdnc

        class ProcDerived(mdnc.data.preprocs.ProcAbstract):
            def __init__(self, a, parent=None):
                super().__init__(parent=parent, _disable_inds=True)
                self.a = a

            def preprocess(self, x, y, z): # All inputs are arrays
                return self.a * np.mean((x, y, z), axis=0)

            def postprocess(self, x_m): # The inverse operator
                x = x_m / self.a
                return x_m, x_m, x_m

        proc = ProcDerived(a=2.0)
        x, y, z = np.ones([5, 2]), np.ones([5, 2]), np.zeros([5, 2])
        xm = proc.preprocess(x, y, z)
        xr, yr, zr = proc.postprocess(xm)
        print('Processed shape:', xm.shape)
        print('Processed error:', np.amax(np.abs(xm - 2 * np.mean([x, y, z], axis=0)))
        print('Inverse error:', np.amax(np.abs(xm - xr)), np.amax(np.abs(xm - yr)), np.amax(np.abs(xm - zr)))
        ```

This operation is not invertible. We could find that the inverse error would be greater than `#!py 0`.

All derived classes of this class could be cascaded with each other. See the tutorial for checking more examples.
