# modules.conv.encnet*

:codicons-symbol-method: Function · [:material-graph-outline: nn.Module][torch-module]

```python
net = encnet*(
    order=2, kernel_size=3, in_planes=1, out_length=2
)
```

Instant presents of `mdnc.module.conv.EncoderNet*d`.

## Arguments

**Requries**

| Argument {: .w-6rem} | Type {: .w-4rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `order` | `#!py int` | The order of the convolutional layers, could be `#!py 1`, `#!py 2`, or `#!py 3`. |
| `kernel_size` | `#!py int` or<br>`#!py (int,)` | The kernel size of each convolutional layer. |
| `in_planes` | `#!py int` | The channel number of the input data. |
| `out_length` | `#!py int` | The length of the output vector, if not set, the output would not be flattened. |

## APIs

| API {: .w-5rem} | Net depth {: .w-4rem} | Channel {: .w-4rem} | Layer configs {: .w-8rem} | Source {: .w-4rem} |
| :-----: | :-------: | :-----: | :-----------: | :-----: |
| `encnet12` | 5 | 64 | `#!py [2, 2, 2, 2, 2]` | [:octicons-file-code-24:]({{ source.root }}/modules/conv.py#L1239){ target="_blank" } |
| `encnet15` | 5 | 64 | `#!py [2, 2, 3, 3, 3]` | [:octicons-file-code-24:]({{ source.root }}/modules/conv.py#L1255){ target="_blank" } |
| `encnet17` | 5 | 64 | `#!py [3, 3, 3, 3, 3]` | [:octicons-file-code-24:]({{ source.root }}/modules/conv.py#L1271){ target="_blank" } |
| `encnet22` | 5 | 64 | `#!py [4, 4, 4, 4, 4]` | [:octicons-file-code-24:]({{ source.root }}/modules/conv.py#L1287){ target="_blank" } |

[torch-module]:https://pytorch.org/docs/stable/generated/torch.nn.Module.html "torch.nn.Module"
