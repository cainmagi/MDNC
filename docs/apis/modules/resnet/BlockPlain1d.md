# modules.resnet.BlockPlain1d

:codicons-symbol-class: Class · [:material-graph-outline: nn.Module][torch-module] · [:octicons-file-code-24: Source]({{ source.root }}/modules/resnet.py#L798){ target="_blank" }

```python
layer = mdnc.modules.resnet.BlockPlain1d(
    in_planes, out_planes,
    kernel_size=3, stride=1, padding=1, output_size=None,
    normalizer='pinst', activator='prelu', layer_order='new', scaler='down'
)
```

In the following paper, the authors propose two structres of the residual block.

[:fontawesome-regular-file-pdf: Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

This is the implementation of the plain (first-type) residual block. The residual block could be divided into two branches (input + conv). In this plain implementation, the convolutional branch is a composed of double convolutional layers. Shown in the following chart:

```mermaid
flowchart TB
    in((" ")) --> conv1[Modern<br>convolution] --> conv2[Modern<br>convolution] --> plus(("+")):::diagramop --> out((" "))
    in --> plus
    classDef diagramop fill:#FFB11B, stroke:#AF811B;
```

If the channel of the output changes, or the size of the output changes, a projection layer implemented by a convolution with `#!py kernel_size=1` is required for mapping the input branch to the output space:

```mermaid
flowchart TB
    in((" ")) --> conv1[Modern<br>convolution] --> conv2[Modern<br>convolution] --> plus(("+")):::diagramop --> out((" "))
    in --> pconv[Projection<br>convolution] --> plus
    classDef diagramop fill:#FFB11B, stroke:#AF811B;
```

In the following paper, a new op composition order is proposed for building residual block:

[:fontawesome-regular-file-pdf: Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)

This implementation called "pre-activation" would change the order of the sub-layers in the modern convolutional layer (see [`mdnc.modules.conv.ConvModern1d`](../../ConvModern1d)). We support and recommend to use this implementation, set `#!py layer_order = 'new'` to enable it.

## Arguments

**Requries**

| Argument {: .w-6rem} | Type {: .w-4rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `in_planes` | `#!py int` | The channel number of the input data. |
| `out_planes` | `#!py int` | The channel number of the output data. |
| `kernel_size` | `#!py int` | The kernel size of this layer. |
| `stride` | `#!py int` | The stride size of this layer. When `#!py scaler='down'`, this argument serves as the down-sampling factor. When `#!py scaler='up'`, this argument serves as the up-sampling factor. |
| `padding` | `#!py int` | The padding size of this layer. The zero padding would be performed on both edges of the input before the convolution. |
| `output_size` | `#!py int` | The length of the output data. This option is only used when `#!py scaler='up'`. When setting this value, the size of the up-sampling would be given explicitly and the argument `stride` would not be used. |
| `normalizer` | `#!py str` | The normalization method, could be: <ul> <li>`#!py 'batch'`: Batch normalization.</li> <li>`#!py 'inst'`: Instance normalization.</li> <li>`#!py 'pinst'`: Instance normalization with tunable rescaling parameters.</li> <li>`#!py 'null'`: Without normalization, would falls back to the "convolution + activation" form. In this case, the `#!py layer_order='new'` would not take effects.</li> </ul> |
| `activator` | `#!py str` | The activation method, could be: `#!py 'prelu'`, `#!py 'relu'`, `#!py 'null'`. |
| `layer_order` | `#!py str` | The sub-layer composition order, could be: <ul> <li>`#!py 'new'`: normalization + activation + convolution.</li> <li>`#!py 'old'`: convolution + normalization + activation.</li> </ul> |
| `scaler` | `#!py str` | The scaling method, could be: <ul> <li>`#!py 'down'`: the argument `stride` would be used for down-sampling.</li> <li>`#!py 'up'`: the argument `stride` would be used for up-sampling (equivalent to transposed convolution).</li> </ul> |

## Operators

### :codicons-symbol-operator: `#!py __call__`

```python
y = layer(x)
```

The forward operator implemented by the `forward()` method. The input is a 1D tensor, and the output is the final output of this layer.

**Requries**

| Argument {: .w-5rem} | Type {: .w-6rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `x` | `#!py torch.Tensor` | A 1D tensor, the size should be `#!py (B, C, L)`, where `B` is the batch size, `C` is the input channel number, and `L` is the input data length. |

**Returns**

| Argument {: .w-5rem} | Description {: .w-8rem} |
| :------: | :---------- |
| `y` | A 1D tensor, the size should be `#!py (B, C, L)`, where `B` is the batch size, `C` is the output channel number, and `L` is the output data length. |

## Examples

In the first example, we build a plain residual block with 1/2 down-sampling and same padding.

???+ example "Example 1"
    === "Codes"
        ```python linenums="1"
        import mdnc

        layer = mdnc.modules.resnet.BlockPlain1d(16, 32, kernel_size=3, stride=2, padding=1, scaler='down')
        mdnc.contribs.torchsummary.summary(layer, (16, 255), device='cpu')
        ```

    === "Output"
        ```
        ----------------------------------------------------------------
                Layer (type)               Output Shape         Param #
        ================================================================
            InstanceNorm1d-1              [-1, 16, 255]              32
                     PReLU-2              [-1, 16, 255]              16
                    Conv1d-3              [-1, 16, 255]             768
            InstanceNorm1d-4              [-1, 16, 255]              32
                     PReLU-5              [-1, 16, 255]              16
                    Conv1d-6              [-1, 32, 128]           1,536
                    Conv1d-7              [-1, 32, 128]             512
            InstanceNorm1d-8              [-1, 32, 128]              64
              BlockPlain1d-9              [-1, 32, 128]               0
        ================================================================
        Total params: 2,976
        Trainable params: 2,976
        Non-trainable params: 0
        ----------------------------------------------------------------
        Input size (MB): 0.02
        Forward/backward pass size (MB): 0.28
        Params size (MB): 0.01
        Estimated Total Size (MB): 0.31
        ----------------------------------------------------------------
        ```

Note that the output length would be `128` in this example, because the same padding is used for the input. In this case, if we want to make a reverse layer, we could specify the `output_size` for the up-sampling layer, for example:

???+ example "Example 2"
    === "Codes"
        ```python linenums="1"
        import mdnc

        layer = mdnc.modules.resnet.BlockPlain1d(32, 16, kernel_size=3, output_size=255, padding=1, scaler='up')
        mdnc.contribs.torchsummary.summary(layer, (32, 128), device='cpu')
        ```

    === "Output"
        ```
        ----------------------------------------------------------------
                Layer (type)               Output Shape         Param #
        ================================================================
            InstanceNorm1d-1              [-1, 32, 128]              64
                     PReLU-2              [-1, 32, 128]              32
                    Conv1d-3              [-1, 32, 128]           3,072
            InstanceNorm1d-4              [-1, 32, 128]              64
                     PReLU-5              [-1, 32, 128]              32
                  Upsample-6              [-1, 32, 255]               0
                    Conv1d-7              [-1, 16, 255]           1,536
                  Upsample-8              [-1, 32, 255]               0
                    Conv1d-9              [-1, 16, 255]             512
           InstanceNorm1d-10              [-1, 16, 255]              32
             BlockPlain1d-11              [-1, 16, 255]               0
        ================================================================
        Total params: 5,344
        Trainable params: 5,344
        Non-trainable params: 0
        ----------------------------------------------------------------
        Input size (MB): 0.02
        Forward/backward pass size (MB): 0.41
        Params size (MB): 0.02
        Estimated Total Size (MB): 0.44
        ----------------------------------------------------------------
        ```

[torch-module]:https://pytorch.org/docs/stable/generated/torch.nn.Module.html "torch.nn.Module"
