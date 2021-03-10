# utils.draw.plot_training_records

:codicons-symbol-method: Function · [:octicons-file-code-24: Source]({{ source.root }}/contribs/torchsummary.py#L58)

```python
mdnc.utils.draw.plot_training_records(
    gen,
    xlabel=None, ylabel='value', x_mark_num=None, y_log=False,
    figure_size=(6, 5.5), legend_loc=None, legend_col=None,
    fig=None, ax=None
)
```

Plot a training curve graph for multiple data groups. Each group is given by:

* 4 1D arrays, representing the x axis of training metrics, the trainining metric values, the x axis of validation metrics, the validation metric values respectively.
* or 2 2D arrays. Both of them have a shape of `#!py (N, 2)`. The two arrays represents the x axis and training metrics, the x axis and validation metric values respectively.
* or 2 1D arrays. In this case, the validation data is not provided. The two arrays represents the x axis of training metrics, the trainining metric values repspectively.
* or a 4D array. The 4 columns represents the x axis of training metrics, the trainining metric values, the x axis of validation metrics, the validation metric values respectively.
* or a 2D array. In this case, the validation data is not provided. The two columns represents the x axis of training metrics, the trainining metric values repspectively.
* or a 1D array. In this case, the validation data is not provided. The data represnets the training metrics. The x axis would be generated automatically.

## Arguments

**Requries**

| Argument {: .w-6rem} | Type {: .w-7rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `gen` | `#!py object` | A generator callable object (function), each `#!py yield` returns a sample. It allows users to provide an extra kwargs dict for each iteration (see [Examples](#examples)). For each iteration, it returns 4 1D arrays, or 2 2D arrays, or 2 1D arrays, or a 4D array, or a 2D array, or a 1D array. |
| `xlabel` | `#!py str`  | The x axis label. |
| `ylabel` | `#!py str`  | The y axis label. |
| `x_mark_num` | `#!py int`  | The number of markers for the x axis. |
| `y_log` | `#!py bool`  | A flag. Whether to convert the y axis into the logarithmic format. |
| `figure_size` | `#!py (float, float)`  | A tuple with two values representing the (width, height) of the output figure. The unit is inch. |
| `legend_loc` | `#!py str` or<br>`#!py int` or<br>`#!py (float, float)` | The localtion of the legend, see [:fontawesome-solid-external-link-alt: matplotlib.pyplot.legend][mpl-legend] to view details. (The legend only works when passing `label` to each iteration). |
| `legend_col` | `#!py int` | The number of columns of the legend, see [:fontawesome-solid-external-link-alt: matplotlib.pyplot.legend][mpl-legend] to view details. (The legend only works when passing `label` to each iteration). |
| `fig` | `#!py object` | A `matplotlib` figure instance. If not given, would use `#!py plt.gcf()` for instead. |
| `ax`  | `#!py object` | A `matplotlib` subplot instance. If not given, would use `#!py plt.gca()` for instead. |

## Examples

???+ example
    === "Codes"
        ```python linenums="1"
        import numpy as np
        import matplotlib.pyplot as plt
        import mdnc

        @mdnc.utils.draw.setFigure(style='Solarize_Light2', font_size=14)
        def test_training_records():
            def func_gen_batch():
                size = 100
                x = np.arange(start=0, stop=size)
                for i in range(3):
                    begin = 1 + 99.0 * np.random.rand()
                    end = 2 + 10 * np.random.rand()
                    v = begin * np.exp((np.square((x - size) / size) - 1.0) * end)
                    yield x, v, {'label': r'$x_{' + str(i + 1) + r'}$'}

            def func_gen_epoch():
                size = 10
                x = np.arange(start=0, stop=size)
                for i in range(3):
                    begin = 1 + 99.0 * np.random.rand()
                    end = 2 + 10 * np.random.rand()
                    v = begin * np.exp((np.square((x - size) / size) - 1.0) * end)
                    val_v = begin * np.exp((np.square((x - size) / size) - 1.0) * (end - 1))
                    data = np.stack([x, v, x, val_v], axis=0)
                    yield data, {'label': r'$x_{' + str(i + 1) + r'}$'}

            mdnc.utils.draw.plot_training_records(func_gen_batch(), y_log=True, x_mark_num=10,
                                                  xlabel='Step', ylabel=r'Batch $\mathcal{L}$')
            plt.show()
            mdnc.utils.draw.plot_training_records(func_gen_epoch(), y_log=True, x_mark_num=10,
                                                  xlabel='Step', ylabel=r'Epoch $\mathcal{L}$')
            plt.show()

        test_training_records()
        ```

[mpl-legend]:https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html "matplotlib.pyplot.legend"
