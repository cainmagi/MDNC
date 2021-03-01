# contribs.torchsummary.summary

Function

```python
summary_str, params_info = mdnc.contribs.torchsummary.summary_str(
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
| `summary_str` | the summary text report. |
| `params_info` | A `#!py tuple` of two values. The first value is the total parameter numbers. The second value is the trainable parameter numbers. |

## Example

See the example of [`mdnc.contribs.torchsummary.summary`](../summary/#example)

???+ tip
    This function could be used for generating the text log file:
    ```python linenums="1"
    ...
    with open('my_module.log', 'w') as f:
        report, _ = mdnc.contribs.torchsummary.summary_string(model, ...)
        f.write(report)
    ```
