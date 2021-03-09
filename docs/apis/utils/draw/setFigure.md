# utils.draw.setFigure

:codicons-symbol-class: Class · :octicons-mention-24: Decorator · :codicons-symbol-field: Context · [:octicons-file-code-24: Source]({{ source.root }}/data/h5py.py#L1351)

```python
mdnc.utils.draw.setFigure(
    style=None, font_name=None, font_size=None, use_tex=None
)
```

A context decorator class, which is used for changing the figure's configurations locally for a specific function. Could be used by two different ways:

???+ example
    === "As decorator"

        ```python linenums="1"
        @mdnc.utils.draw.setFigure(font_size=12)
        def plot_curve():
            plot_figures1(...)
            plot_figures2(...)
            ...
        ```

    === "As context"

        ```python linenums="1"
        with mdnc.utils.draw.setFigure(font_size=12):
            plot_figures1(...)
            plot_figures2(...)
            ...
        ```

## Arguments

**Requries**

The following arguments would take effect only when they are configured explicitly.

| Argument {: .w-5rem} | Type {: .w-5rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `style` | `#!py str` or<br>`#!py dict` or<br>`#!py Path` or<br>`#!py list` | The local stylesheet of the figure. The details could be found in [:fontawesome-solid-external-link-alt: matplotlib.style.context][mpl-stylecontext]. We could also find some examples [:fontawesome-solid-external-link-alt: here][mpl-styleeg]. |
| `font_name` | `#!py str` | The local font family name for the output figure. The specified font should be installed and available for any software. |
| `font_size` | `#!py int` | The local font size for the output figure. |
| `use_tex` | `#!py bool` | Whether to use LaTeX backend for the output figure. Recommend to enable it when drawing figures for a paper. |

??? warning
    If you are using a not installed or not supported font as `font_name`, the context decorator would not raise an error, but only show some warnings. This behavior is the same as `matplotlib`.

??? warning
    In the above argument list, the latter argument would override the former argument. For example, if some `style` has already specified `font_size`, configuring the argument `font_size` would override the configuration in the stylesheet. Please pay attention to your desired configurations.

## Examples

???+ example "As decorator"
    === "Codes"
        ```python linenums="1"
        import numpy as np
        import matplotlib.pyplot as plt
        import mdnc

        @mdnc.utils.draw.setFigure(style='classic', font_size=16, font_name='arial')
        def plot_local_setting():
            t = np.linspace(-10, 10, 100)
            plt.plot(t, 1 / (1 + np.exp(-t)))
            plt.title('In the context, font: arial.')
            plt.show()

        plot_local_setting()
        ```

???+ example "As context"
    === "Codes"
        ```python linenums="1"
        import numpy as np
        import matplotlib.pyplot as plt
        import mdnc

        with mdnc.utils.draw.setFigure(style='classic', font_size=16, font_name='arial'):
            t = np.linspace(-10, 10, 100)
            plt.plot(t, 1 / (1 + np.exp(-t)))
            plt.title('In the context, font: arial.')
            plt.show()
        ```

[mpl-stylecontext]:https://matplotlib.org/stable/api/style_api.html?highlight=style%20context#matplotlib.style.context
[mpl-styleeg]:https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
