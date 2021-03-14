# modules.conv.unet*

:codicons-symbol-method: Function · [:material-graph-outline: nn.Module][torch-module]

```python
net = unet*(
    order=2, kernel_size=3, in_planes=1, out_planes=1
)
```

Instant presents of `mdnc.module.conv.UNet*d`.

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
| `unet12` | 3 | 64 | `#!py [2, 2, 2]` | [:octicons-file-code-24:]({{ source.root }}/modules/conv.py#L1075){ target="_blank" } |
| `unet16` | 4 | 64 | `#!py [2, 2, 2, 2]` | [:octicons-file-code-24:]({{ source.root }}/modules/conv.py#L1091){ target="_blank" } |
| `unet17` | 3 | 64 | `#!py [3, 3, 3]` | [:octicons-file-code-24:]({{ source.root }}/modules/conv.py#L1107){ target="_blank" } |
| `unet23` | 4 | 64 | `#!py [3, 3, 3, 3]` | [:octicons-file-code-24:]({{ source.root }}/modules/conv.py#L1123){ target="_blank" } |
| `unet29` | 5 | 64 | `#!py [3, 3, 3, 3, 3]` | [:octicons-file-code-24:]({{ source.root }}/modules/conv.py#L1139){ target="_blank" } |

where `unet29` is a nearly replicated work of @milesial/Pytorch-UNet. The only difference of `unet29` is two extra convolutional layers for the input and output mapping.

[torch-module]:https://pytorch.org/docs/stable/generated/torch.nn.Module.html "torch.nn.Module"
