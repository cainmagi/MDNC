# modules.resnet.unet*

:codicons-symbol-method: Function · [:material-graph-outline: nn.Module][torch-module]

```python
net = unet*(
    order=2, kernel_size=3, in_planes=1, out_planes=1
)
```

Instant presents of `mdnc.module.resnet.UNet*d`.

## Arguments

**Requries**

| Argument {: .w-6rem} | Type {: .w-4rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `order` | `#!py int` | The order of the residual blocks, could be `#!py 1`, `#!py 2`, or `#!py 3`. |
| `kernel_size` | `#!py int` or<br>`#!py (int,)` | The kernel size of each residual block. |
| `in_planes` | `#!py int` | The channel number of the input data. |
| `out_planes` | `#!py int` | The channel number of the output data. |

## APIs

| API {: .w-5rem} | Net depth {: .w-4rem} | Channel {: .w-4rem} | Block type {: .w-5rem} | Layer configs {: .w-8rem} | Source {: .w-4rem} |
| :-----: | :-------: | :-----: | :--------: | :-----------: | :-----: |
| `unet16` | 3 | 64 | `plain`      | `#!py [2, 1, 1]` | [:octicons-file-code-24:]({{ source.root }}/modules/resnet.py#L1467){ target="_blank" } |
| `unet32` | 3 | 64 | `bottleneck` | `#!py [2, 2, 2]` | [:octicons-file-code-24:]({{ source.root }}/modules/resnet.py#L1486){ target="_blank" } |
| `unet44` | 4 | 64 | `bottleneck` | `#!py [2, 2, 2, 2]` | [:octicons-file-code-24:]({{ source.root }}/modules/resnet.py#L1503){ target="_blank" } |
| `unet65` | 4 | 64 | `bottleneck` | `#!py [3, 3, 3, 3]` | [:octicons-file-code-24:]({{ source.root }}/modules/resnet.py#L1520){ target="_blank" } |
| `unet83` | 5 | 64 | `bottleneck` | `#!py [3, 3, 3, 3, 3]` | [:octicons-file-code-24:]({{ source.root }}/modules/resnet.py#L1537){ target="_blank" } |

where `unet16` is a nearly replicated work of @nikhilroxtomar/Deep-Residual-Unet. The only difference of `unet16` is one more modern convolutional layer in the first stage.

[torch-module]:https://pytorch.org/docs/stable/generated/torch.nn.Module.html "torch.nn.Module"
