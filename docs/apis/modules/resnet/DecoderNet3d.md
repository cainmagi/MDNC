# modules.resnet.DecoderNet3d

:codicons-symbol-class: Class · [:material-graph-outline: nn.Module][torch-module] · [:octicons-file-code-24: Source]({{ source.root }}/modules/resnet.py#L1391){ target="_blank" }

```python
net = mdnc.modules.resnet.DecoderNet3d(
     channel, layers, out_size, block='bottleneck',
     kernel_size=3, in_length=2, out_planes=1
)
```

This moule is a built-in model for 3D residual decoder network. This network could be used as a part of the auto-encoder, or just a network for up-sampling (or generating) data.

The network would up-sample the input data according to the network depth. The depth is given by the length of the argument `layers`.  The network structure is shown in the following chart:

```mermaid
flowchart TB
    u1["Block 1<br>Stack of layers[0] blocks"]
    u2["Block 2<br>Stack of layers[1] blocks"]
    ui["Block ...<br>Stack of ... blocks"]
    un["Block n<br>Stack of layers[n-1] blocks"]
    optional:::blockoptional
    subgraph optional [Optional]
       cin["Conv3d<br>with unsqueeze"]
    end
    u1 -->|up<br>sampling| u2 -->|up<br>sampling| ui -->|up<br>sampling| un
    cin -.-> u1
    linkStyle 0,1,2 stroke-width:4px, stroke:#080 ;
    classDef blockoptional fill:none, stroke-dasharray:10,10, stroke:#9370DB, width:100;
```

The argument `layers` is a sequence of `#!py int`. For each block $i$, it contains `#!py layers[i-1]` repeated residual blocks (see [`mdnc.modules.resnet.BlockPlain3d`](../BlockPlain3d) and [`mdnc.modules.resnet.BlockBottleneck3d`](../BlockBottleneck3d)). Each down-sampling or up-sampling is configured by `#!py stride=2`. The channel number would be doubled in the up-sampling route. An optional unsqueezer and convolutional layer could be prepended to the first layer when the argument `#!py in_length != None`. This optional layer is used for converting the vector features in initial feature maps.

## Arguments

**Requries**

| Argument {: .w-6rem} | Type {: .w-7rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `channel` | `#!py int` | The channel number of the first hidden block (layer). After each down-sampling, the channel number would be doubled. |
| `layers` | `#!py (int,)` | A sequence of layer numbers for each block. Each number represents the number of residual blocks of a stage (block). The stage numer, i.e. the depth of the network is the length of this list. |
| `out_size` | `#!py int` or<br>`#!py (int, int, int)` | The size of the output data. This argument needs to be specified by users, because the network needs to configure its layers according to the output size. |
| `block` | `#!py str` | The residual block type, could be: <ul> <li>`#!py 'plain'`: see [`BlockPlain3d`](../BlockPlain3d).</li> <li>`#!py 'bottleneck'`: see [`BlockBottleneck3d`](../BlockBottleneck3d).</li> </ul> |
| `kernel_size` | `#!py int` or<br>`#!py (int, int, int)` | The kernel size of each residual block. |
| `in_length` | `#!py int` | The length of the input vector, if not set, the input needs to be feature maps. See the property [`input_size`](#input_size) to check the input data size in this case. |
| `out_planes` | `#!py int` | The channel number of the output data. |

## Operators

### :codicons-symbol-operator: `#!py __call__`

```python
y = net(x)
```

The forward operator implemented by the `forward()` method. The input data is a tensor with a size determined by configurations. The output is a 3D tensor. The channel number of the output is specified by the argument `out_planes`.

**Requries**

| Argument {: .w-5rem} | Type {: .w-6rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `x` | `#!py torch.Tensor` | A tensor, <ul> <li>When `#!py in_length is None`: the size should be `#!py (B, L)`, where `B` is the batch size, and `L` is `in_length`.</li> <li>When `#!py in_length != None`: the size should be `#!py (B, C, L1, L2, L3)`, where `B` is the batch size, `C` and `(L1, L2, L3)` are the channel number and the size of the input feature maps (see [`input_size`](#input_size)) respectively.</li> </ul> |

**Returns**

| Argument {: .w-5rem} | Description {: .w-8rem} |
| :------: | :---------- |
| `y` | A 3D tensor, the size should be `#!py (B, C, L1, L2, L3)`, where `B` is the batch size, `C` is the input channel number, and `(L1, L2, L3)` is the output data size specified by the argument `out_size`. |

## Properties

### :codicons-symbol-variable: `nlayers`

```python
net.nlayers
```

The total number of convolutional layers along the depth of the network. This value would not take the fully-connected layer into consideration.

-----

### :codicons-symbol-variable: `input_size`

```python
net.input_size
```

The size of the input data size (a `#!py tuple`). This property is useful when `#!py in_length is None`. In this case, the input size is determined by the network.

??? warning
    This size contains the channel number (as the first element), because the input channel number is also determined by network when `#!py in_length is None`.

## Examples

???+ example "Example 1"
    === "Codes"
        ```python linenums="1"
        import mdnc

        net = mdnc.modules.resnet.DecoderNet3d(64, [2, 2, 2, 2, 2], in_length=32, out_size=(31, 32, 30), out_planes=3)
        print('The number of convolutional layers along the depth is {0}.'.format(net.nlayers))
        print('The input size is {0}.'.format(net.input_size))
        mdnc.contribs.torchsummary.summary(net, net.input_size, device='cpu')
        ```

    === "Output"
        ```
        The number of convolutional layers along the depth is 33.
        The input size is (32,).
        ----------------------------------------------------------------
                Layer (type)               Output Shape         Param #
        ================================================================
                    Conv3d-1        [-1, 1024, 1, 1, 1]          33,792
                    Conv3d-2        [-1, 1024, 1, 1, 1]      28,311,552
            InstanceNorm3d-3        [-1, 1024, 1, 1, 1]           2,048
                     PReLU-4        [-1, 1024, 1, 1, 1]           1,024
                    Conv3d-5        [-1, 1024, 1, 1, 1]       1,048,576
            InstanceNorm3d-6        [-1, 1024, 1, 1, 1]           2,048
                     PReLU-7        [-1, 1024, 1, 1, 1]           1,024
                    Conv3d-8        [-1, 1024, 1, 1, 1]      28,311,552
            InstanceNorm3d-9        [-1, 1024, 1, 1, 1]           2,048
                    PReLU-10        [-1, 1024, 1, 1, 1]           1,024
                   Conv3d-11         [-1, 512, 1, 1, 1]         524,288
                   Conv3d-12         [-1, 512, 1, 1, 1]         524,288
           InstanceNorm3d-13         [-1, 512, 1, 1, 1]           1,024
        _BlockBo...neckNd-14         [-1, 512, 1, 1, 1]               0
           InstanceNorm3d-15         [-1, 512, 1, 1, 1]           1,024
                    PReLU-16         [-1, 512, 1, 1, 1]             512
                   Conv3d-17         [-1, 512, 1, 1, 1]         262,144
           InstanceNorm3d-18         [-1, 512, 1, 1, 1]           1,024
                    PReLU-19         [-1, 512, 1, 1, 1]             512
                 Upsample-20         [-1, 512, 2, 2, 2]               0
                   Conv3d-21         [-1, 512, 2, 2, 2]       7,077,888
           InstanceNorm3d-22         [-1, 512, 2, 2, 2]           1,024
                    PReLU-23         [-1, 512, 2, 2, 2]             512
                   Conv3d-24         [-1, 512, 2, 2, 2]         262,144
                 Upsample-25         [-1, 512, 2, 2, 2]               0
                   Conv3d-26         [-1, 512, 2, 2, 2]         262,144
           InstanceNorm3d-27         [-1, 512, 2, 2, 2]           1,024
        _BlockBo...neckNd-28         [-1, 512, 2, 2, 2]               0
           _BlockResStkNd-29         [-1, 512, 2, 2, 2]               0
           InstanceNorm3d-30         [-1, 512, 2, 2, 2]           1,024
                    PReLU-31         [-1, 512, 2, 2, 2]             512
                   Conv3d-32         [-1, 512, 2, 2, 2]         262,144
           InstanceNorm3d-33         [-1, 512, 2, 2, 2]           1,024
                    PReLU-34         [-1, 512, 2, 2, 2]             512
                   Conv3d-35         [-1, 512, 2, 2, 2]       7,077,888
           InstanceNorm3d-36         [-1, 512, 2, 2, 2]           1,024
                    PReLU-37         [-1, 512, 2, 2, 2]             512
                   Conv3d-38         [-1, 256, 2, 2, 2]         131,072
                   Conv3d-39         [-1, 256, 2, 2, 2]         131,072
           InstanceNorm3d-40         [-1, 256, 2, 2, 2]             512
        _BlockBo...neckNd-41         [-1, 256, 2, 2, 2]               0
           InstanceNorm3d-42         [-1, 256, 2, 2, 2]             512
                    PReLU-43         [-1, 256, 2, 2, 2]             256
                   Conv3d-44         [-1, 256, 2, 2, 2]          65,536
           InstanceNorm3d-45         [-1, 256, 2, 2, 2]             512
                    PReLU-46         [-1, 256, 2, 2, 2]             256
                 Upsample-47         [-1, 256, 4, 4, 4]               0
                   Conv3d-48         [-1, 256, 4, 4, 4]       1,769,472
           InstanceNorm3d-49         [-1, 256, 4, 4, 4]             512
                    PReLU-50         [-1, 256, 4, 4, 4]             256
                   Conv3d-51         [-1, 256, 4, 4, 4]          65,536
                 Upsample-52         [-1, 256, 4, 4, 4]               0
                   Conv3d-53         [-1, 256, 4, 4, 4]          65,536
           InstanceNorm3d-54         [-1, 256, 4, 4, 4]             512
        _BlockBo...neckNd-55         [-1, 256, 4, 4, 4]               0
           _BlockResStkNd-56         [-1, 256, 4, 4, 4]               0
           InstanceNorm3d-57         [-1, 256, 4, 4, 4]             512
                    PReLU-58         [-1, 256, 4, 4, 4]             256
                   Conv3d-59         [-1, 256, 4, 4, 4]          65,536
           InstanceNorm3d-60         [-1, 256, 4, 4, 4]             512
                    PReLU-61         [-1, 256, 4, 4, 4]             256
                   Conv3d-62         [-1, 256, 4, 4, 4]       1,769,472
           InstanceNorm3d-63         [-1, 256, 4, 4, 4]             512
                    PReLU-64         [-1, 256, 4, 4, 4]             256
                   Conv3d-65         [-1, 128, 4, 4, 4]          32,768
                   Conv3d-66         [-1, 128, 4, 4, 4]          32,768
           InstanceNorm3d-67         [-1, 128, 4, 4, 4]             256
        _BlockBo...neckNd-68         [-1, 128, 4, 4, 4]               0
           InstanceNorm3d-69         [-1, 128, 4, 4, 4]             256
                    PReLU-70         [-1, 128, 4, 4, 4]             128
                   Conv3d-71         [-1, 128, 4, 4, 4]          16,384
           InstanceNorm3d-72         [-1, 128, 4, 4, 4]             256
                    PReLU-73         [-1, 128, 4, 4, 4]             128
                 Upsample-74         [-1, 128, 8, 8, 8]               0
                   Conv3d-75         [-1, 128, 8, 8, 8]         442,368
           InstanceNorm3d-76         [-1, 128, 8, 8, 8]             256
                    PReLU-77         [-1, 128, 8, 8, 8]             128
                   Conv3d-78         [-1, 128, 8, 8, 8]          16,384
                 Upsample-79         [-1, 128, 8, 8, 8]               0
                   Conv3d-80         [-1, 128, 8, 8, 8]          16,384
           InstanceNorm3d-81         [-1, 128, 8, 8, 8]             256
        _BlockBo...neckNd-82         [-1, 128, 8, 8, 8]               0
           _BlockResStkNd-83         [-1, 128, 8, 8, 8]               0
           InstanceNorm3d-84         [-1, 128, 8, 8, 8]             256
                    PReLU-85         [-1, 128, 8, 8, 8]             128
                   Conv3d-86         [-1, 128, 8, 8, 8]          16,384
           InstanceNorm3d-87         [-1, 128, 8, 8, 8]             256
                    PReLU-88         [-1, 128, 8, 8, 8]             128
                   Conv3d-89         [-1, 128, 8, 8, 8]         442,368
           InstanceNorm3d-90         [-1, 128, 8, 8, 8]             256
                    PReLU-91         [-1, 128, 8, 8, 8]             128
                   Conv3d-92          [-1, 64, 8, 8, 8]           8,192
                   Conv3d-93          [-1, 64, 8, 8, 8]           8,192
           InstanceNorm3d-94          [-1, 64, 8, 8, 8]             128
        _BlockBo...neckNd-95          [-1, 64, 8, 8, 8]               0
           InstanceNorm3d-96          [-1, 64, 8, 8, 8]             128
                    PReLU-97          [-1, 64, 8, 8, 8]              64
                   Conv3d-98          [-1, 64, 8, 8, 8]           4,096
           InstanceNorm3d-99          [-1, 64, 8, 8, 8]             128
                   PReLU-100          [-1, 64, 8, 8, 8]              64
                Upsample-101       [-1, 64, 16, 16, 16]               0
                  Conv3d-102       [-1, 64, 16, 16, 16]         110,592
          InstanceNorm3d-103       [-1, 64, 16, 16, 16]             128
                   PReLU-104       [-1, 64, 16, 16, 16]              64
                  Conv3d-105       [-1, 64, 16, 16, 16]           4,096
                Upsample-106       [-1, 64, 16, 16, 16]               0
                  Conv3d-107       [-1, 64, 16, 16, 16]           4,096
          InstanceNorm3d-108       [-1, 64, 16, 16, 16]             128
        _BlockBo...eckNd-109       [-1, 64, 16, 16, 16]               0
          _BlockResStkNd-110       [-1, 64, 16, 16, 16]               0
          InstanceNorm3d-111       [-1, 64, 16, 16, 15]             128
                   PReLU-112       [-1, 64, 16, 16, 15]              64
                  Conv3d-113       [-1, 64, 16, 16, 15]           4,096
          InstanceNorm3d-114       [-1, 64, 16, 16, 15]             128
                   PReLU-115       [-1, 64, 16, 16, 15]              64
                  Conv3d-116       [-1, 64, 16, 16, 15]         110,592
          InstanceNorm3d-117       [-1, 64, 16, 16, 15]             128
                   PReLU-118       [-1, 64, 16, 16, 15]              64
                  Conv3d-119       [-1, 64, 16, 16, 15]           4,096
        _BlockBo...eckNd-120       [-1, 64, 16, 16, 15]               0
          InstanceNorm3d-121       [-1, 64, 16, 16, 15]             128
                   PReLU-122       [-1, 64, 16, 16, 15]              64
                  Conv3d-123       [-1, 64, 16, 16, 15]           4,096
          InstanceNorm3d-124       [-1, 64, 16, 16, 15]             128
                   PReLU-125       [-1, 64, 16, 16, 15]              64
                Upsample-126       [-1, 64, 32, 32, 30]               0
                  Conv3d-127       [-1, 64, 32, 32, 30]         110,592
          InstanceNorm3d-128       [-1, 64, 32, 32, 30]             128
                   PReLU-129       [-1, 64, 32, 32, 30]              64
                  Conv3d-130       [-1, 64, 32, 32, 30]           4,096
                Upsample-131       [-1, 64, 32, 32, 30]               0
                  Conv3d-132       [-1, 64, 32, 32, 30]           4,096
          InstanceNorm3d-133       [-1, 64, 32, 32, 30]             128
        _BlockBo...eckNd-134       [-1, 64, 32, 32, 30]               0
          _BlockResStkNd-135       [-1, 64, 32, 32, 30]               0
                  Conv3d-136        [-1, 3, 31, 32, 30]          24,003
            DecoderNet3d-137        [-1, 3, 31, 32, 30]               0
        ================================================================
        Total params: 79,473,411
        Trainable params: 79,473,411
        Non-trainable params: 0
        ----------------------------------------------------------------
        Input size (MB): 0.00
        Forward/backward pass size (MB): 214.37
        Params size (MB): 303.17
        Estimated Total Size (MB): 517.53
        ----------------------------------------------------------------
        ```

???+ example "Example 2"
    === "Codes"
        ```python linenums="1"
        import mdnc

        net = mdnc.modules.resnet.DecoderNet3d(64, [2, 2, 2, 2, 2], in_length=None, out_size=(31, 32, 30), out_planes=3)
        print('The number of convolutional layers along the depth is {0}.'.format(net.nlayers))
        print('The input size is {0}.'.format(net.input_size))
        mdnc.contribs.torchsummary.summary(net, net.input_size, device='cpu')
        ```

    === "Output"
        ```
        The number of convolutional layers along the depth is 32.
        The input size is (1024, 1, 1, 1).
        ----------------------------------------------------------------
                Layer (type)               Output Shape         Param #
        ================================================================
                    Conv3d-1        [-1, 1024, 1, 1, 1]      28,311,552
            InstanceNorm3d-2        [-1, 1024, 1, 1, 1]           2,048
                     PReLU-3        [-1, 1024, 1, 1, 1]           1,024
                    Conv3d-4        [-1, 1024, 1, 1, 1]       1,048,576
            InstanceNorm3d-5        [-1, 1024, 1, 1, 1]           2,048
                     PReLU-6        [-1, 1024, 1, 1, 1]           1,024
                    Conv3d-7        [-1, 1024, 1, 1, 1]      28,311,552
            InstanceNorm3d-8        [-1, 1024, 1, 1, 1]           2,048
                     PReLU-9        [-1, 1024, 1, 1, 1]           1,024
                   Conv3d-10         [-1, 512, 1, 1, 1]         524,288
                   Conv3d-11         [-1, 512, 1, 1, 1]         524,288
           InstanceNorm3d-12         [-1, 512, 1, 1, 1]           1,024
        _BlockBo...neckNd-13         [-1, 512, 1, 1, 1]               0
           InstanceNorm3d-14         [-1, 512, 1, 1, 1]           1,024
                    PReLU-15         [-1, 512, 1, 1, 1]             512
                   Conv3d-16         [-1, 512, 1, 1, 1]         262,144
           InstanceNorm3d-17         [-1, 512, 1, 1, 1]           1,024
                    PReLU-18         [-1, 512, 1, 1, 1]             512
                 Upsample-19         [-1, 512, 2, 2, 2]               0
                   Conv3d-20         [-1, 512, 2, 2, 2]       7,077,888
           InstanceNorm3d-21         [-1, 512, 2, 2, 2]           1,024
                    PReLU-22         [-1, 512, 2, 2, 2]             512
                   Conv3d-23         [-1, 512, 2, 2, 2]         262,144
                 Upsample-24         [-1, 512, 2, 2, 2]               0
                   Conv3d-25         [-1, 512, 2, 2, 2]         262,144
           InstanceNorm3d-26         [-1, 512, 2, 2, 2]           1,024
        _BlockBo...neckNd-27         [-1, 512, 2, 2, 2]               0
           _BlockResStkNd-28         [-1, 512, 2, 2, 2]               0
           InstanceNorm3d-29         [-1, 512, 2, 2, 2]           1,024
                    PReLU-30         [-1, 512, 2, 2, 2]             512
                   Conv3d-31         [-1, 512, 2, 2, 2]         262,144
           InstanceNorm3d-32         [-1, 512, 2, 2, 2]           1,024
                    PReLU-33         [-1, 512, 2, 2, 2]             512
                   Conv3d-34         [-1, 512, 2, 2, 2]       7,077,888
           InstanceNorm3d-35         [-1, 512, 2, 2, 2]           1,024
                    PReLU-36         [-1, 512, 2, 2, 2]             512
                   Conv3d-37         [-1, 256, 2, 2, 2]         131,072
                   Conv3d-38         [-1, 256, 2, 2, 2]         131,072
           InstanceNorm3d-39         [-1, 256, 2, 2, 2]             512
        _BlockBo...neckNd-40         [-1, 256, 2, 2, 2]               0
           InstanceNorm3d-41         [-1, 256, 2, 2, 2]             512
                    PReLU-42         [-1, 256, 2, 2, 2]             256
                   Conv3d-43         [-1, 256, 2, 2, 2]          65,536
           InstanceNorm3d-44         [-1, 256, 2, 2, 2]             512
                    PReLU-45         [-1, 256, 2, 2, 2]             256
                 Upsample-46         [-1, 256, 4, 4, 4]               0
                   Conv3d-47         [-1, 256, 4, 4, 4]       1,769,472
           InstanceNorm3d-48         [-1, 256, 4, 4, 4]             512
                    PReLU-49         [-1, 256, 4, 4, 4]             256
                   Conv3d-50         [-1, 256, 4, 4, 4]          65,536
                 Upsample-51         [-1, 256, 4, 4, 4]               0
                   Conv3d-52         [-1, 256, 4, 4, 4]          65,536
           InstanceNorm3d-53         [-1, 256, 4, 4, 4]             512
        _BlockBo...neckNd-54         [-1, 256, 4, 4, 4]               0
           _BlockResStkNd-55         [-1, 256, 4, 4, 4]               0
           InstanceNorm3d-56         [-1, 256, 4, 4, 4]             512
                    PReLU-57         [-1, 256, 4, 4, 4]             256
                   Conv3d-58         [-1, 256, 4, 4, 4]          65,536
           InstanceNorm3d-59         [-1, 256, 4, 4, 4]             512
                    PReLU-60         [-1, 256, 4, 4, 4]             256
                   Conv3d-61         [-1, 256, 4, 4, 4]       1,769,472
           InstanceNorm3d-62         [-1, 256, 4, 4, 4]             512
                    PReLU-63         [-1, 256, 4, 4, 4]             256
                   Conv3d-64         [-1, 128, 4, 4, 4]          32,768
                   Conv3d-65         [-1, 128, 4, 4, 4]          32,768
           InstanceNorm3d-66         [-1, 128, 4, 4, 4]             256
        _BlockBo...neckNd-67         [-1, 128, 4, 4, 4]               0
           InstanceNorm3d-68         [-1, 128, 4, 4, 4]             256
                    PReLU-69         [-1, 128, 4, 4, 4]             128
                   Conv3d-70         [-1, 128, 4, 4, 4]          16,384
           InstanceNorm3d-71         [-1, 128, 4, 4, 4]             256
                    PReLU-72         [-1, 128, 4, 4, 4]             128
                 Upsample-73         [-1, 128, 8, 8, 8]               0
                   Conv3d-74         [-1, 128, 8, 8, 8]         442,368
           InstanceNorm3d-75         [-1, 128, 8, 8, 8]             256
                    PReLU-76         [-1, 128, 8, 8, 8]             128
                   Conv3d-77         [-1, 128, 8, 8, 8]          16,384
                 Upsample-78         [-1, 128, 8, 8, 8]               0
                   Conv3d-79         [-1, 128, 8, 8, 8]          16,384
           InstanceNorm3d-80         [-1, 128, 8, 8, 8]             256
        _BlockBo...neckNd-81         [-1, 128, 8, 8, 8]               0
           _BlockResStkNd-82         [-1, 128, 8, 8, 8]               0
           InstanceNorm3d-83         [-1, 128, 8, 8, 8]             256
                    PReLU-84         [-1, 128, 8, 8, 8]             128
                   Conv3d-85         [-1, 128, 8, 8, 8]          16,384
           InstanceNorm3d-86         [-1, 128, 8, 8, 8]             256
                    PReLU-87         [-1, 128, 8, 8, 8]             128
                   Conv3d-88         [-1, 128, 8, 8, 8]         442,368
           InstanceNorm3d-89         [-1, 128, 8, 8, 8]             256
                    PReLU-90         [-1, 128, 8, 8, 8]             128
                   Conv3d-91          [-1, 64, 8, 8, 8]           8,192
                   Conv3d-92          [-1, 64, 8, 8, 8]           8,192
           InstanceNorm3d-93          [-1, 64, 8, 8, 8]             128
        _BlockBo...neckNd-94          [-1, 64, 8, 8, 8]               0
           InstanceNorm3d-95          [-1, 64, 8, 8, 8]             128
                    PReLU-96          [-1, 64, 8, 8, 8]              64
                   Conv3d-97          [-1, 64, 8, 8, 8]           4,096
           InstanceNorm3d-98          [-1, 64, 8, 8, 8]             128
                    PReLU-99          [-1, 64, 8, 8, 8]              64
                Upsample-100       [-1, 64, 16, 16, 16]               0
                  Conv3d-101       [-1, 64, 16, 16, 16]         110,592
          InstanceNorm3d-102       [-1, 64, 16, 16, 16]             128
                   PReLU-103       [-1, 64, 16, 16, 16]              64
                  Conv3d-104       [-1, 64, 16, 16, 16]           4,096
                Upsample-105       [-1, 64, 16, 16, 16]               0
                  Conv3d-106       [-1, 64, 16, 16, 16]           4,096
          InstanceNorm3d-107       [-1, 64, 16, 16, 16]             128
        _BlockBo...eckNd-108       [-1, 64, 16, 16, 16]               0
          _BlockResStkNd-109       [-1, 64, 16, 16, 16]               0
          InstanceNorm3d-110       [-1, 64, 16, 16, 15]             128
                   PReLU-111       [-1, 64, 16, 16, 15]              64
                  Conv3d-112       [-1, 64, 16, 16, 15]           4,096
          InstanceNorm3d-113       [-1, 64, 16, 16, 15]             128
                   PReLU-114       [-1, 64, 16, 16, 15]              64
                  Conv3d-115       [-1, 64, 16, 16, 15]         110,592
          InstanceNorm3d-116       [-1, 64, 16, 16, 15]             128
                   PReLU-117       [-1, 64, 16, 16, 15]              64
                  Conv3d-118       [-1, 64, 16, 16, 15]           4,096
        _BlockBo...eckNd-119       [-1, 64, 16, 16, 15]               0
          InstanceNorm3d-120       [-1, 64, 16, 16, 15]             128
                   PReLU-121       [-1, 64, 16, 16, 15]              64
                  Conv3d-122       [-1, 64, 16, 16, 15]           4,096
          InstanceNorm3d-123       [-1, 64, 16, 16, 15]             128
                   PReLU-124       [-1, 64, 16, 16, 15]              64
                Upsample-125       [-1, 64, 32, 32, 30]               0
                  Conv3d-126       [-1, 64, 32, 32, 30]         110,592
          InstanceNorm3d-127       [-1, 64, 32, 32, 30]             128
                   PReLU-128       [-1, 64, 32, 32, 30]              64
                  Conv3d-129       [-1, 64, 32, 32, 30]           4,096
                Upsample-130       [-1, 64, 32, 32, 30]               0
                  Conv3d-131       [-1, 64, 32, 32, 30]           4,096
          InstanceNorm3d-132       [-1, 64, 32, 32, 30]             128
        _BlockBo...eckNd-133       [-1, 64, 32, 32, 30]               0
          _BlockResStkNd-134       [-1, 64, 32, 32, 30]               0
                  Conv3d-135        [-1, 3, 31, 32, 30]          24,003
            DecoderNet3d-136        [-1, 3, 31, 32, 30]               0
        ================================================================
        Total params: 79,439,619
        Trainable params: 79,439,619
        Non-trainable params: 0
        ----------------------------------------------------------------
        Input size (MB): 0.00
        Forward/backward pass size (MB): 214.36
        Params size (MB): 303.04
        Estimated Total Size (MB): 517.40
        ----------------------------------------------------------------
        ```

[torch-module]:https://pytorch.org/docs/stable/generated/torch.nn.Module.html "torch.nn.Module"
