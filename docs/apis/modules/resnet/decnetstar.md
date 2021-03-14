# modules.resnet.decnet*

:codicons-symbol-method: Function · [:material-graph-outline: nn.Module][torch-module]

```python
net = decnet*(
    out_size, order=2, kernel_size=3, in_length=2, out_planes=1
)
```

Instant presents of `mdnc.module.resnet.DecoderNet*d`.

## Arguments

**Requries**

| Argument {: .w-6rem} | Type {: .w-4rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `out_size` | `#!py int` or<br>`#!py (int,)` | The size of the output data. This argument needs to be specified by users, because the network needs to configure its layers according to the output size. |
| `order` | `#!py int` | The order of the residual blocks, could be `#!py 1`, `#!py 2`, or `#!py 3`. |
| `kernel_size` | `#!py int` or<br>`#!py (int,)` | The kernel size of each residual block. |
| `in_length` | `#!py int` | The length of the input vector, if not set, the input needs to be feature maps. See the property [`input_size`](#input_size) to check the input data size in this case. |
| `out_planes` | `#!py int` | The channel number of the output data. |

## APIs

| API {: .w-5rem} | Net depth {: .w-4rem} | Channel {: .w-4rem} | Block type {: .w-5rem} | Layer configs {: .w-8rem} | Source {: .w-4rem} |
| :-----: | :-------: | :-----: | :--------: | :-----------: | :-----: |
| `decnet13` | 5 | 64 | `plain`      | `#!py [1, 1, 1, 1, 1]` | [:octicons-file-code-24:]({{ source.root }}/modules/resnet.py#L1710){ target="_blank" } |
| `decnet33` | 5 | 64 | `bottleneck` | `#!py [2, 2, 2, 2, 2]` | [:octicons-file-code-24:]({{ source.root }}/modules/resnet.py#L1728){ target="_blank" } |
| `decnet48` | 5 | 64 | `bottleneck` | `#!py [3, 3, 3, 3, 3]` | [:octicons-file-code-24:]({{ source.root }}/modules/resnet.py#L1746){ target="_blank" } |
| `decnet63` | 5 | 64 | `bottleneck` | `#!py [4, 4, 4, 4, 4]` | [:octicons-file-code-24:]({{ source.root }}/modules/resnet.py#L1764){ target="_blank" } |

[torch-module]:https://pytorch.org/docs/stable/generated/torch.nn.Module.html "torch.nn.Module"
