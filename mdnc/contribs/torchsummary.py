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
#   1. Fix the bug of parameter number calculation when there
#      are more than one variables.
#   2. Make multuple output variables split into multiple lines.
#   3. Remove the last line break of summary_string()
#   4. Enable argument "device" to accept both str and
#      torch.device.
################################################################
'''

import collections
import numpy as np

import torch
import torch.nn as nn


def summary(model, input_size, batch_size=-1, device='cuda:0', dtypes=None):
    if isinstance(device, str):
        device = torch.device(device)
    result, params_info = summary_string(
        model, input_size, batch_size, device, dtypes)
    print(result)

    return params_info


def summary_string(model, input_size, batch_size=-1, device='cuda:0', dtypes=None):
    if isinstance(device, str):
        device = torch.device(device)

    if dtypes is None:
        dtypes = [torch.FloatTensor] * len(input_size)

    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = collections.OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList)):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

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
        # input_shape, output_shape, trainable, nb_params
        sum_layer = summary[layer]
        if len(sum_layer["output_shape"]) > 0 and isinstance(sum_layer["output_shape"][0], (list, tuple)):  # Add multiple output support
            line_new = ["{:>20}  {:>25} {:>15}".format(
                layer,
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
                layer,
                str(sum_layer["output_shape"]),
                "{0:,}".format(sum_layer["nb_params"]),
            )
        total_params += sum_layer["nb_params"]

        output_shape = sum_layer["output_shape"]
        if isinstance(output_shape[0], (list, tuple)):
            total_output += np.sum(list(map(np.prod, output_shape)), dtype=np.int)
        else:
            total_output += np.prod(output_shape, dtype=np.int)
        if "trainable" in sum_layer:
            if sum_layer["trainable"] is True:
                trainable_params += sum_layer["nb_params"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.sum(list(map(np.prod, input_size))) * batch_size * 4. / (1024 ** 2.))
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
