# contribs.torchsummary.summary

:codicons-symbol-method: Function · [:octicons-file-code-24: Source]({{ source.root }}/contribs/torchsummary.py#L58)

```python
params_info = mdnc.contribs.torchsummary.summary(
    model, input_size, batch_size=-1, device='cuda:0', dtypes=None
)
```

Iterate the whole pytorch model and summarize the infomation as a Keras-style text report. The output would be store in a str.

## Arguments

**Requries**

| Argument {: .w-6rem} | Type {: .w-8rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `model`  | `#!py nn.Module` | The pyTorch network module instance. It is to be analyzed. |
| `input_size` | `#!py (seq/int, )` | A sequence (`#!py list/tuple`) or a sequence of sequnces, indicating the size of the each model input variable. |
| `batch_size` | `#!py int` | The batch size used for testing and displaying the results. |
| `device` | `#!py str` or<br>`#!py torch.device` | Should be set according to the deployed device of the argument `model`. |
| `dtypes` | `#!py (torch.dtype, )` | A sequence of torch data type for each input variable. If set `#!py None`, would use float type for all variables. |

**Returns**

| Argument {: .w-6rem} | Description {: .w-8rem} |
| :------: | :---------- |
| `params_info` | A `#!py tuple` of two values. The first value is the total parameter numbers. The second value is the trainable parameter numbers. |

## Examples

???+ example
    === "Codes"
        ```python linenums="1"
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import mdnc

        class TestTupleOutModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1a = nn.Linear(300, 50)
                self.fc1b = nn.Linear(50, 10)

                self.fc2a = nn.Linear(300, 50)
                self.fc2b = nn.Linear(50, 10)

            def forward(self, x1, x2):
                x1 = F.relu(self.fc1a(x1))
                x1 = self.fc1b(x1)
                x2 = x2.type(torch.FloatTensor)
                x2 = F.relu(self.fc2a(x2))
                x2 = self.fc2b(x2)
                # set x2 to FloatTensor
                x = torch.cat((x1, x2), 0)
                return F.log_softmax(x, dim=1), F.log_softmax(x1, dim=1), F.log_softmax(x2, dim=1)

        input1 = (1, 300)
        input2 = (1, 300)
        dtypes = (torch.FloatTensor, torch.LongTensor)
        total_params, trainable_params = mdnc.contribs.torchsummary.summary(
            TestTupleOutModule(), (input1, input2), device='cpu', dtypes=dtypes)
        ```

    === "Output"
        ```
        ----------------------------------------------------------------
                Layer (type)               Output Shape         Param #
        ================================================================
                    Linear-1                [-1, 1, 50]          15,050
                    Linear-2                [-1, 1, 10]             510
                    Linear-3                [-1, 1, 50]          15,050
                    Linear-4                [-1, 1, 10]             510
        TestTupleOutModule-5                [-1, 1, 10]               0
                                            [-1, 1, 10]                
                                            [-1, 1, 10]                
        ================================================================
        Total params: 31,120
        Trainable params: 31,120
        Non-trainable params: 0
        ----------------------------------------------------------------
        Input size (MB): 0.00
        Forward/backward pass size (MB): 0.00
        Params size (MB): 0.12
        Estimated Total Size (MB): 0.12
        ----------------------------------------------------------------
        ```
