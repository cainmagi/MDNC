# modules.resnet.UNet1d

:codicons-symbol-class: Class · [:material-graph-outline: nn.Module][torch-module] · [:octicons-file-code-24: Source]({{ source.root }}/modules/resnet.py#L1086){ target="_blank" }

```python
net = mdnc.modules.resnet.UNet1d(
    channel, layers, block='bottleneck',
    kernel_size=3, in_planes=1, out_planes=1
)
```

This moule is a built-in model for 1D residual U-Net. The network is inspired by:

@nikhilroxtomar/Deep-Residual-Unet

The network would down-sample and up-sample the input data according to the network depth. The depth is given by the length of the argument `layers`.  The network structure is shown in the following chart:

```mermaid
flowchart TB
    b1["Block 1<br>Stack of layers[0] blocks"]
    b2["Block 2<br>Stack of layers[1] blocks"]
    bi["Block ...<br>Stack of ... blocks"]
    bn["Block n<br>Stack of layers[n-1] blocks"]
    u1["Block 2n-1<br>Stack of layers[0] blocks"]
    u2["Block 2n-2<br>Stack of layers[1] blocks"]
    ui["Block ...<br>Stack of ... blocks"]
    b1 -->|down<br>sampling| b2 -->|down<br>sampling| bi -->|down<br>sampling| bn
    bn -->|up<br>sampling| ui -->|up<br>sampling| u2 -->|up<br>sampling| u1
    b1 -->|skip<br>connection| u1
    b2 -->|skip<br>connection| u2
    bi -->|skip<br>connection| ui
    linkStyle 0,1,2 stroke-width:4px, stroke:#800 ;
    linkStyle 3,4,5 stroke-width:4px, stroke:#080 ;
    linkStyle 6,7,8 stroke-width:4px, stroke:#888 ;
```

The argument `layers` is a sequence of `#!py int`. For each block $i$, it contains `#!py layers[i-1]` repeated residual blocks (see [`mdnc.modules.resnet.BlockPlain1d`](../BlockPlain1d) and [`mdnc.modules.resnet.BlockBottleneck1d`](../BlockBottleneck1d)). Each down-sampling or up-sampling is configured by `#!py stride=2`. The channel number would be doubled in the down-sampling route and reduced to 1/2 in the up-sampling route. The skip connection is perfromed by concatenation.

## Arguments

**Requries**

| Argument {: .w-6rem} | Type {: .w-4rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `channel` | `#!py int` | The channel number of the first hidden block (layer). After each down-sampling, the channel number would be doubled. After each up-sampling, the channel number would be reduced to 1/2. |
| `layers` | `#!py (int,)` | A sequence of layer numbers for each block. Each number represents the number of residual blocks of a stage (block). The stage numer, i.e. the depth of the network is the length of this list. |
| `block` | `#!py str` | The residual block type, could be: <ul> <li>`#!py 'plain'`: see [`BlockPlain1d`](../BlockPlain1d).</li> <li>`#!py 'bottleneck'`: see [`BlockBottleneck1d`](../BlockBottleneck1d).</li> </ul> |
| `kernel_size` | `#!py int` | The kernel size of each residual block. |
| `in_planes` | `#!py int` | The channel number of the input data. |
| `out_planes` | `#!py int` | The channel number of the output data. |

## Operators

### :codicons-symbol-operator: `#!py __call__`

```python
y = net(x)
```

The forward operator implemented by the `forward()` method. The input is a 1D tensor, and the output is the final output of this network.

**Requries**

| Argument {: .w-5rem} | Type {: .w-6rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `x` | `#!py torch.Tensor` | A 1D tensor, the size should be `#!py (B, C, L)`, where `B` is the batch size, `C` is the input channel number, and `L` is the input data length. |

**Returns**

| Argument {: .w-5rem} | Description {: .w-8rem} |
| :------: | :---------- |
| `y` | A 1D tensor, the size should be `#!py (B, C, L)`, where `B` is the batch size, `C` is the output channel number, and `L` is the **input** data length. |

## Properties

### :codicons-symbol-variable: `nlayers`

```python
net.nlayers
```

The total number of convolutional layers along the depth of the network.

## Examples

???+ example "Example"
    === "Codes"
        ```python linenums="1"
        import mdnc

        net = mdnc.modules.resnet.UNet1d(64, [2, 2, 2, 2, 3], in_planes=3, out_planes=1)
        print('The number of convolutional layers along the depth is {0}.'.format(net.nlayers))
        mdnc.contribs.torchsummary.summary(net, (3, 128), device='cpu')
        ```

    === "Output"
        ```
        The number of convolutional layers along the depth is 59.
        ----------------------------------------------------------------
                Layer (type)               Output Shape         Param #
        ================================================================
                    Conv1d-1              [-1, 64, 128]             960
            InstanceNorm1d-2              [-1, 64, 128]             128
                     PReLU-3              [-1, 64, 128]              64
                    Conv1d-4              [-1, 64, 128]           4,096
            InstanceNorm1d-5              [-1, 64, 128]             128
                     PReLU-6              [-1, 64, 128]              64
                    Conv1d-7              [-1, 64, 128]          12,288
            InstanceNorm1d-8              [-1, 64, 128]             128
                     PReLU-9              [-1, 64, 128]              64
                   Conv1d-10              [-1, 64, 128]           4,096
        _BlockBo...neckNd-11              [-1, 64, 128]               0
           InstanceNorm1d-12              [-1, 64, 128]             128
                    PReLU-13              [-1, 64, 128]              64
                   Conv1d-14              [-1, 64, 128]           4,096
           InstanceNorm1d-15              [-1, 64, 128]             128
                    PReLU-16              [-1, 64, 128]              64
                   Conv1d-17               [-1, 64, 64]          12,288
           InstanceNorm1d-18               [-1, 64, 64]             128
                    PReLU-19               [-1, 64, 64]              64
                   Conv1d-20               [-1, 64, 64]           4,096
                   Conv1d-21               [-1, 64, 64]           4,096
           InstanceNorm1d-22               [-1, 64, 64]             128
        _BlockBo...neckNd-23               [-1, 64, 64]               0
           _BlockResStkNd-24               [-1, 64, 64]               0
                                          [-1, 64, 128]
           InstanceNorm1d-25               [-1, 64, 64]             128
                    PReLU-26               [-1, 64, 64]              64
                   Conv1d-27               [-1, 64, 64]           4,096
           InstanceNorm1d-28               [-1, 64, 64]             128
                    PReLU-29               [-1, 64, 64]              64
                   Conv1d-30               [-1, 64, 64]          12,288
           InstanceNorm1d-31               [-1, 64, 64]             128
                    PReLU-32               [-1, 64, 64]              64
                   Conv1d-33              [-1, 128, 64]           8,192
                   Conv1d-34              [-1, 128, 64]           8,192
           InstanceNorm1d-35              [-1, 128, 64]             256
        _BlockBo...neckNd-36              [-1, 128, 64]               0
           InstanceNorm1d-37              [-1, 128, 64]             256
                    PReLU-38              [-1, 128, 64]             128
                   Conv1d-39              [-1, 128, 64]          16,384
           InstanceNorm1d-40              [-1, 128, 64]             256
                    PReLU-41              [-1, 128, 64]             128
                   Conv1d-42              [-1, 128, 32]          49,152
           InstanceNorm1d-43              [-1, 128, 32]             256
                    PReLU-44              [-1, 128, 32]             128
                   Conv1d-45              [-1, 128, 32]          16,384
                   Conv1d-46              [-1, 128, 32]          16,384
           InstanceNorm1d-47              [-1, 128, 32]             256
        _BlockBo...neckNd-48              [-1, 128, 32]               0
           _BlockResStkNd-49              [-1, 128, 32]               0
                                          [-1, 128, 64]
           InstanceNorm1d-50              [-1, 128, 32]             256
                    PReLU-51              [-1, 128, 32]             128
                   Conv1d-52              [-1, 128, 32]          16,384
           InstanceNorm1d-53              [-1, 128, 32]             256
                    PReLU-54              [-1, 128, 32]             128
                   Conv1d-55              [-1, 128, 32]          49,152
           InstanceNorm1d-56              [-1, 128, 32]             256
                    PReLU-57              [-1, 128, 32]             128
                   Conv1d-58              [-1, 256, 32]          32,768
                   Conv1d-59              [-1, 256, 32]          32,768
           InstanceNorm1d-60              [-1, 256, 32]             512
        _BlockBo...neckNd-61              [-1, 256, 32]               0
           InstanceNorm1d-62              [-1, 256, 32]             512
                    PReLU-63              [-1, 256, 32]             256
                   Conv1d-64              [-1, 256, 32]          65,536
           InstanceNorm1d-65              [-1, 256, 32]             512
                    PReLU-66              [-1, 256, 32]             256
                   Conv1d-67              [-1, 256, 16]         196,608
           InstanceNorm1d-68              [-1, 256, 16]             512
                    PReLU-69              [-1, 256, 16]             256
                   Conv1d-70              [-1, 256, 16]          65,536
                   Conv1d-71              [-1, 256, 16]          65,536
           InstanceNorm1d-72              [-1, 256, 16]             512
        _BlockBo...neckNd-73              [-1, 256, 16]               0
           _BlockResStkNd-74              [-1, 256, 16]               0
                                          [-1, 256, 32]
           InstanceNorm1d-75              [-1, 256, 16]             512
                    PReLU-76              [-1, 256, 16]             256
                   Conv1d-77              [-1, 256, 16]          65,536
           InstanceNorm1d-78              [-1, 256, 16]             512
                    PReLU-79              [-1, 256, 16]             256
                   Conv1d-80              [-1, 256, 16]         196,608
           InstanceNorm1d-81              [-1, 256, 16]             512
                    PReLU-82              [-1, 256, 16]             256
                   Conv1d-83              [-1, 512, 16]         131,072
                   Conv1d-84              [-1, 512, 16]         131,072
           InstanceNorm1d-85              [-1, 512, 16]           1,024
        _BlockBo...neckNd-86              [-1, 512, 16]               0
           InstanceNorm1d-87              [-1, 512, 16]           1,024
                    PReLU-88              [-1, 512, 16]             512
                   Conv1d-89              [-1, 512, 16]         262,144
           InstanceNorm1d-90              [-1, 512, 16]           1,024
                    PReLU-91              [-1, 512, 16]             512
                   Conv1d-92               [-1, 512, 8]         786,432
           InstanceNorm1d-93               [-1, 512, 8]           1,024
                    PReLU-94               [-1, 512, 8]             512
                   Conv1d-95               [-1, 512, 8]         262,144
                   Conv1d-96               [-1, 512, 8]         262,144
           InstanceNorm1d-97               [-1, 512, 8]           1,024
        _BlockBo...neckNd-98               [-1, 512, 8]               0
           _BlockResStkNd-99               [-1, 512, 8]               0
                                          [-1, 512, 16]
          InstanceNorm1d-100               [-1, 512, 8]           1,024
                   PReLU-101               [-1, 512, 8]             512
                  Conv1d-102               [-1, 512, 8]         262,144
          InstanceNorm1d-103               [-1, 512, 8]           1,024
                   PReLU-104               [-1, 512, 8]             512
                  Conv1d-105               [-1, 512, 8]         786,432
          InstanceNorm1d-106               [-1, 512, 8]           1,024
                   PReLU-107               [-1, 512, 8]             512
                  Conv1d-108              [-1, 1024, 8]         524,288
                  Conv1d-109              [-1, 1024, 8]         524,288
          InstanceNorm1d-110              [-1, 1024, 8]           2,048
        _BlockBo...eckNd-111              [-1, 1024, 8]               0
          InstanceNorm1d-112              [-1, 1024, 8]           2,048
                   PReLU-113              [-1, 1024, 8]           1,024
                  Conv1d-114              [-1, 1024, 8]       1,048,576
          InstanceNorm1d-115              [-1, 1024, 8]           2,048
                   PReLU-116              [-1, 1024, 8]           1,024
                  Conv1d-117              [-1, 1024, 8]       3,145,728
          InstanceNorm1d-118              [-1, 1024, 8]           2,048
                   PReLU-119              [-1, 1024, 8]           1,024
                  Conv1d-120              [-1, 1024, 8]       1,048,576
        _BlockBo...eckNd-121              [-1, 1024, 8]               0
          InstanceNorm1d-122              [-1, 1024, 8]           2,048
                   PReLU-123              [-1, 1024, 8]           1,024
                  Conv1d-124              [-1, 1024, 8]       1,048,576
          InstanceNorm1d-125              [-1, 1024, 8]           2,048
                   PReLU-126              [-1, 1024, 8]           1,024
                Upsample-127             [-1, 1024, 16]               0
                  Conv1d-128             [-1, 1024, 16]       3,145,728
          InstanceNorm1d-129             [-1, 1024, 16]           2,048
                   PReLU-130             [-1, 1024, 16]           1,024
                  Conv1d-131              [-1, 512, 16]         524,288
                Upsample-132             [-1, 1024, 16]               0
                  Conv1d-133              [-1, 512, 16]         524,288
          InstanceNorm1d-134              [-1, 512, 16]           1,024
        _BlockBo...eckNd-135              [-1, 512, 16]               0
          _BlockResStkNd-136              [-1, 512, 16]               0
          InstanceNorm1d-137             [-1, 1024, 16]           2,048
                   PReLU-138             [-1, 1024, 16]           1,024
                  Conv1d-139             [-1, 1024, 16]       1,048,576
          InstanceNorm1d-140             [-1, 1024, 16]           2,048
                   PReLU-141             [-1, 1024, 16]           1,024
                  Conv1d-142             [-1, 1024, 16]       3,145,728
          InstanceNorm1d-143             [-1, 1024, 16]           2,048
                   PReLU-144             [-1, 1024, 16]           1,024
                  Conv1d-145              [-1, 512, 16]         524,288
                  Conv1d-146              [-1, 512, 16]         524,288
          InstanceNorm1d-147              [-1, 512, 16]           1,024
        _BlockBo...eckNd-148              [-1, 512, 16]               0
          InstanceNorm1d-149              [-1, 512, 16]           1,024
                   PReLU-150              [-1, 512, 16]             512
                  Conv1d-151              [-1, 512, 16]         262,144
          InstanceNorm1d-152              [-1, 512, 16]           1,024
                   PReLU-153              [-1, 512, 16]             512
                Upsample-154              [-1, 512, 32]               0
                  Conv1d-155              [-1, 512, 32]         786,432
          InstanceNorm1d-156              [-1, 512, 32]           1,024
                   PReLU-157              [-1, 512, 32]             512
                  Conv1d-158              [-1, 256, 32]         131,072
                Upsample-159              [-1, 512, 32]               0
                  Conv1d-160              [-1, 256, 32]         131,072
          InstanceNorm1d-161              [-1, 256, 32]             512
        _BlockBo...eckNd-162              [-1, 256, 32]               0
          _BlockResStkNd-163              [-1, 256, 32]               0
          InstanceNorm1d-164              [-1, 512, 32]           1,024
                   PReLU-165              [-1, 512, 32]             512
                  Conv1d-166              [-1, 512, 32]         262,144
          InstanceNorm1d-167              [-1, 512, 32]           1,024
                   PReLU-168              [-1, 512, 32]             512
                  Conv1d-169              [-1, 512, 32]         786,432
          InstanceNorm1d-170              [-1, 512, 32]           1,024
                   PReLU-171              [-1, 512, 32]             512
                  Conv1d-172              [-1, 256, 32]         131,072
                  Conv1d-173              [-1, 256, 32]         131,072
          InstanceNorm1d-174              [-1, 256, 32]             512
        _BlockBo...eckNd-175              [-1, 256, 32]               0
          InstanceNorm1d-176              [-1, 256, 32]             512
                   PReLU-177              [-1, 256, 32]             256
                  Conv1d-178              [-1, 256, 32]          65,536
          InstanceNorm1d-179              [-1, 256, 32]             512
                   PReLU-180              [-1, 256, 32]             256
                Upsample-181              [-1, 256, 64]               0
                  Conv1d-182              [-1, 256, 64]         196,608
          InstanceNorm1d-183              [-1, 256, 64]             512
                   PReLU-184              [-1, 256, 64]             256
                  Conv1d-185              [-1, 128, 64]          32,768
                Upsample-186              [-1, 256, 64]               0
                  Conv1d-187              [-1, 128, 64]          32,768
          InstanceNorm1d-188              [-1, 128, 64]             256
        _BlockBo...eckNd-189              [-1, 128, 64]               0
          _BlockResStkNd-190              [-1, 128, 64]               0
          InstanceNorm1d-191              [-1, 256, 64]             512
                   PReLU-192              [-1, 256, 64]             256
                  Conv1d-193              [-1, 256, 64]          65,536
          InstanceNorm1d-194              [-1, 256, 64]             512
                   PReLU-195              [-1, 256, 64]             256
                  Conv1d-196              [-1, 256, 64]         196,608
          InstanceNorm1d-197              [-1, 256, 64]             512
                   PReLU-198              [-1, 256, 64]             256
                  Conv1d-199              [-1, 128, 64]          32,768
                  Conv1d-200              [-1, 128, 64]          32,768
          InstanceNorm1d-201              [-1, 128, 64]             256
        _BlockBo...eckNd-202              [-1, 128, 64]               0
          InstanceNorm1d-203              [-1, 128, 64]             256
                   PReLU-204              [-1, 128, 64]             128
                  Conv1d-205              [-1, 128, 64]          16,384
          InstanceNorm1d-206              [-1, 128, 64]             256
                   PReLU-207              [-1, 128, 64]             128
                Upsample-208             [-1, 128, 128]               0
                  Conv1d-209             [-1, 128, 128]          49,152
          InstanceNorm1d-210             [-1, 128, 128]             256
                   PReLU-211             [-1, 128, 128]             128
                  Conv1d-212              [-1, 64, 128]           8,192
                Upsample-213             [-1, 128, 128]               0
                  Conv1d-214              [-1, 64, 128]           8,192
          InstanceNorm1d-215              [-1, 64, 128]             128
        _BlockBo...eckNd-216              [-1, 64, 128]               0
          _BlockResStkNd-217              [-1, 64, 128]               0
          InstanceNorm1d-218             [-1, 128, 128]             256
                   PReLU-219             [-1, 128, 128]             128
                  Conv1d-220             [-1, 128, 128]          16,384
          InstanceNorm1d-221             [-1, 128, 128]             256
                   PReLU-222             [-1, 128, 128]             128
                  Conv1d-223             [-1, 128, 128]          49,152
          InstanceNorm1d-224             [-1, 128, 128]             256
                   PReLU-225             [-1, 128, 128]             128
                  Conv1d-226              [-1, 64, 128]           8,192
                  Conv1d-227              [-1, 64, 128]           8,192
          InstanceNorm1d-228              [-1, 64, 128]             128
        _BlockBo...eckNd-229              [-1, 64, 128]               0
          InstanceNorm1d-230              [-1, 64, 128]             128
                   PReLU-231              [-1, 64, 128]              64
                  Conv1d-232              [-1, 64, 128]           4,096
          InstanceNorm1d-233              [-1, 64, 128]             128
                   PReLU-234              [-1, 64, 128]              64
                  Conv1d-235              [-1, 64, 128]          12,288
          InstanceNorm1d-236              [-1, 64, 128]             128
                   PReLU-237              [-1, 64, 128]              64
                  Conv1d-238              [-1, 64, 128]           4,096
        _BlockBo...eckNd-239              [-1, 64, 128]               0
          _BlockResStkNd-240              [-1, 64, 128]               0
                  Conv1d-241               [-1, 1, 128]             321
                  UNet1d-242               [-1, 1, 128]               0
        ================================================================
        Total params: 24,157,569
        Trainable params: 24,157,569
        Non-trainable params: 0
        ----------------------------------------------------------------
        Input size (MB): 0.00
        Forward/backward pass size (MB): 16.50
        Params size (MB): 92.15
        Estimated Total Size (MB): 108.66
        ----------------------------------------------------------------
        ```

[torch-module]:https://pytorch.org/docs/stable/generated/torch.nn.Module.html "torch.nn.Module"
