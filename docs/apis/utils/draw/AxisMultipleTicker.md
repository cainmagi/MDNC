# utils.draw.AxisMultipleTicker

:codicons-symbol-class: Class · [:octicons-file-code-24: Source]({{ source.root }}/data/h5py.py#L1351)

```python
amticker = mdnc.utils.draw.AxisMultipleTicker(
    den_major=2, den_minor=5, number=np.pi, symbol=r'\pi'
)
```

Use multiple locator to define the formatted axis. Inspired by the following post:

https://stackoverflow.com/a/53586826

This class is a factory class, which could produce 3 properties used for formatting a special axis (like a trigonometric axis): `formatter`, `major_locator`, `minor_locator`. See the [Examples](#examples) to view how to use it.

## Arguments

**Requries**

| Argument {: .w-5rem} | Type {: .w-4rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `den_major` | `#!py int`   | The denominator for the major ticks. |
| `den_minor` | `#!py int`   | The denominator for the minor ticks. |
| `number`    | `#!py float` | The value that each `symbol` represents. |
| `symbol`    | `#!py int`   | The displayed symbol of the major ticks. |

## Properties

### :codicons-symbol-variable: `formatter`

```python
amticker.formatter
```

The major formatter. Use the `matplotlib` API `#!py axis.set_major_formatter()` to set it.

-----

### :codicons-symbol-variable: `major_locator`

```python
amticker.major_locator
```

The major locator. Use the `matplotlib` API `#!py axis.set_major_locator()` to set it.

-----

### :codicons-symbol-variable: `minor_locator`

```python
amticker.minor_locator
```

The minor locator. Use the `matplotlib` API `#!py axis.set_set_minor_locator()` to set it.

## Examples

???+ example
    === "Codes"
        ```python linenums="1"
        import numpy as np
        import matplotlib.pyplot as plt
        import mdnc

        t = np.linspace(-3, 3, 100)
        amticker = mdnc.utils.draw.AxisMultipleTicker(number=np.pi, symbol=r'\pi')
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3))
        ax.plot(t, np.arctan2(np.sin(t * np.pi), np.cos(t * np.pi)))
        ax.yaxis.set_major_formatter(amticker.formatter)
        ax.yaxis.set_major_locator(amticker.major_locator)
        ax.yaxis.set_minor_locator(amticker.minor_locator)
        plt.tight_layout()
        plt.show()
        ```
