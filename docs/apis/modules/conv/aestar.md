# modules.conv.ae*

:codicons-symbol-method: Function · [:material-graph-outline: nn.Module][torch-module]

```python
net = ae*(
    order=2, kernel_size=3, in_planes=1, out_planes=1
)
```

Instant presents of `mdnc.module.conv.AE*d`.

## Arguments

**Requries**

| Argument {: .w-6rem} | Type {: .w-4rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `order` | `#!py int` | The order of the convolutional layers, could be `#!py 1`, `#!py 2`, or `#!py 3`. |
| `kernel_size` | `#!py int` or<br>`#!py (int,)` | The kernel size of each convolutional layer. |
| `in_planes` | `#!py int` | The channel number of the input data. |
| `out_planes` | `#!py int` | The channel number of the output data. |

## APIs

| API {: .w-5rem} | Net depth {: .w-4rem} | Channel {: .w-4rem} | Layer configs {: .w-8rem} | Source {: .w-4rem} |
| :-----: | :-------: | :-----: | :-----------: | :-----: |
| `ae12` | 3 | 64 | `#!py [2, 2, 2]` | [:octicons-file-code-24:]({{ source.root }}/modules/conv.py#L1158){ target="_blank" } |
| `ae16` | 4 | 64 | `#!py [2, 2, 2, 2]` | [:octicons-file-code-24:]({{ source.root }}/modules/conv.py#L1174){ target="_blank" } |
| `ae17` | 3 | 64 | `#!py [3, 3, 3]` | [:octicons-file-code-24:]({{ source.root }}/modules/conv.py#L1190){ target="_blank" } |
| `ae23` | 4 | 64 | `#!py [3, 3, 3, 3]` | [:octicons-file-code-24:]({{ source.root }}/modules/conv.py#L1206){ target="_blank" } |
| `ae29` | 5 | 64 | `#!py [3, 3, 3, 3, 3]` | [:octicons-file-code-24:]({{ source.root }}/modules/conv.py#L1222){ target="_blank" } |

[torch-module]:https://pytorch.org/docs/stable/generated/torch.nn.Module.html "torch.nn.Module"
