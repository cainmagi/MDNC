#!python
# -*- coding: UTF-8 -*-
'''
################################################################
# TorchSummary
# @ Modern Deep Network Toolkits for pyTorch
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.5+
#   numpy 1.13+
#   pyTorch 1.0.0+
# Revised torchsummary lib. The original work could be found
# here:
#   https://github.com/sksq96/pytorch-summary
# The revision includes:
#    1. Fix the bug of parameter number calculation when there
#       is more than one output variables, including both
#       sequence case and dict case.
#    2. Make multiple output variables split into multiple
#       lines.
#    3. Remove the last line break of summary_string()
#    4. Enable argument "device" to accept both str and
#       torch.device.
#    5. Fix a bug when the model requires "batch_size" to be a
#       specific number.
#    6. Fix a bug caused by multiple input cases when
#       "dtypes=None".
#    7. Add text auto wrap when the layer name is too long.
#    8. Support counting all parameters instead of "weight" and
#       "bias".
#    9. Drop the np.sum/prod to fix the overflow problem during
#       calculating the total size.
#   10. Drop the features for gathering the input_shape. This
#       feature is not used during generating the report.
#   11. Add docstring.
################################################################
'''

import functools
import collections

import torch
import torch.nn as nn


def long_sum(v):
    if not all(map(lambda x: isinstance(x, int), v)):
        raise ValueError('contribs.torchsummary: The long_sum only supports the sequence with all int elements.')
    return functools.reduce(lambda x, y: x + y, v)


def long_prod(v):
    if not all(map(lambda x: isinstance(x, int), v)):
        raise ValueError('contribs.torchsummary: The long_sum only supports the sequence with all int elements.')
    return functools.reduce(lambda x, y: x * y, v)


def summary(model, input_size, batch_size=-1, device='cuda:0', dtypes=None):
    '''Keras-style torch summary
    Iterate the whole pytorch model and summarize the infomation as a Keras-style
    text report. The output would be store in a str.
    Arguments:
        model: an instance of nn.Module
        input_size: a sequence (list/tuple) or a sequence of sequnces, indicating
                    the size of the each model input variable.
        batch_size: a int. The batch size used for testing and displaying the
                    results.
        device: a str or torch.device. Should be set according to the deployed
                device of the argument "model".
        dtypes: a list or torch data type for each input variable.
    Returns:
        1. tensor, total parameter numbers.
        2. tensor, trainable parameter numbers.
    '''
    if isinstance(device, str):
        device = torch.device(device)
    result, params_info = summary_string(
        model, input_size, batch_size, device, dtypes)
    print(result)

    return params_info


def summary_string(model, input_size, batch_size=-1, device='cuda:0', dtypes=None):
    '''Keras-style torch summary (string output)
    Iterate the whole pytorch model and summarize the infomation as a Keras-style
    text report. The output would be store in a str.
    Arguments:
        model: an instance of nn.Module
        input_size: a sequence (list/tuple) or a sequence of sequnces, indicating
                    the size of the each model input variable.
        batch_size: a int. The batch size used for testing and displaying the
                    results.
        device: a str or torch.device. Should be set according to the deployed
                device of the argument "model".
        dtypes: a list or torch data type for each input variable.
    Returns:
        1. str, the summary text report.
        2. tuple, (total parameter numbers, trainable parameter numbers)
    '''
    if isinstance(device, str):
        device = torch.device(device)

    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = '{name:s}-{idx:d}'.format(name=class_name, idx=module_idx + 1)
            sum_layer = collections.OrderedDict()
            summary[m_key] = sum_layer
            # The following codes are not actually used, skipped.
            # if isinstance(input[0], dict):
            #     sum_layer["input_shape"] = list(next(iter(input[0].values())).size())
            # else:
            #     sum_layer["input_shape"] = list(input[0].size())
            # sum_layer["input_shape"][0] = batch_size
            if isinstance(output, dict):
                sum_layer["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output.values()
                ]
            elif isinstance(output, (list, tuple)):
                sum_layer["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                sum_layer["output_shape"] = list(output.size())
                sum_layer["output_shape"][0] = batch_size

            params = 0
            params_trainable = 0
            for param in module.parameters(recurse=False):
                nb_param = torch.prod(torch.LongTensor(list(param.size()))).item()
                params += nb_param
                params_trainable += nb_param if param.requires_grad else 0
            sum_layer["nb_params"] = params
            sum_layer["nb_params_trainable"] = params_trainable

        if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList)):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, (list, tuple)) and len(input_size) > 0:
        if not isinstance(input_size[0], (list, tuple)):
            input_size = (input_size, )
    else:
        raise ValueError('contribs.torchsummary: The argument "input_size" is not a tuple of a sequence of tuple. Given "{0}".'.format(input_size))

    if dtypes is None:
        dtypes = [torch.FloatTensor] * len(input_size)
    if len(dtypes) != len(input_size):
        raise ValueError('contribs.torchsummary: The lengths of the arguments "input_size" and "dtypes" does not correspond to each other.')

    # batch_size of 2 for batchnorm
    if batch_size == -1:
        batch_size_ = 2
    else:
        batch_size_ = batch_size
    x = [torch.rand(batch_size_, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = collections.OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    summary_str += "----------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "================================================================" + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # output_shape, trainable, nb_params
        sum_layer = summary[layer]
        if len(layer) > 20:
            layer_disp = '{lhead}...{ltail}'.format(lhead=layer[:8], ltail=layer[-9:])  # 20 = 9 + 8 + 3
        else:
            layer_disp = layer
        if len(sum_layer["output_shape"]) > 0 and isinstance(sum_layer["output_shape"][0], (list, tuple)):  # Add multiple output support
            line_new = ["{:>20}  {:>25} {:>15}".format(
                layer_disp,
                str(sum_layer["output_shape"][0]),
                "{0:,}".format(sum_layer["nb_params"]),
            )]
            for oshape in sum_layer["output_shape"][1:]:
                line_new.append("{:>20}  {:>25} {:>15}".format(
                    '', str(oshape), ''
                ))
            line_new = '\n'.join(line_new)
        else:
            line_new = "{:>20}  {:>25} {:>15}".format(
                layer_disp,
                str(sum_layer["output_shape"]),
                "{0:,}".format(sum_layer["nb_params"]),
            )
        total_params += sum_layer["nb_params"]

        output_shape = sum_layer["output_shape"]
        if isinstance(output_shape[0], (list, tuple)):
            total_output += long_sum(list(map(long_prod, output_shape)))
        else:
            total_output += long_prod(output_shape)
        trainable_params += sum_layer["nb_params_trainable"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(long_sum(list(map(long_prod, input_size))) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "================================================================" + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params - trainable_params) + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "----------------------------------------------------------------"
    # return summary
    return summary_str, (total_params, trainable_params)
