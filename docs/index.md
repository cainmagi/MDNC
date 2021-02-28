# <span style="color:var(--md-primary-fg-color)">M</span>odern <span style="color:var(--md-primary-fg-color)">D</span>eep <span style="color:var(--md-primary-fg-color)">N</span>etwork Toolkits for pyTor<span style="color:var(--md-primary-fg-color)">c</span>h (MDNC)

This is a pyTorch framework used for

* Creating specially designed networks or layers.
* Parallel Data pre- and post- processing as a powerful alternative of `torch.utils.data.DataLoader`.
* Callback-function based data visualizer as an alternative of `seaborn`.
* Web tools for downloading tarball-packed datasets from Github.
* Some modified third-party utilities.

???+ info
    Currently, this module is still under development. The current version is
    [=12% "0.1.2"]
    However, you could use this nightly version anyway. All available (stable) APIs of the current version would be recorded in this document.

## Overview

The pyTorch has its own coding style. In brief, we could summarize the style as below:

* Modules, optimizers, and the training logic are separated from each other. The module is only used for defining the network graph. The optimizers are provided as instances used for loss functions. The training and testing logics require to be implemented by users.
* The data loading and processing are paralleled by `torch.utils.data.DataLoader` and `torchvision` respectively. The users do not need to write codes about multi-processing management.
* The data type conversion requires to be implemented by the users. For example, the predicted variables of the network require to be converted to the correct data type explicitly.
* Because the training logic is implemented by users, arbitrary codes are allowed to be injected during the training loop. Instead of writing callbacks (we do such things when using @keras-team/keras or @scikit-learn/scikit-learn), users could invoke their customized functions (like saving records, showing progress) easily.

This toolkit is designed according to the style. We do not want to make this toolkit look like @keras-team/keras or @PyTorchLightning/pytorch-lightning. In other words, we want to make it special enough for you to use it.

The motivations why we develop this toolkit include:

* Provide simpler interfaces for building more complicated networks, like residual network and DenseNet. The built-in APIs in this toolkit would help users avoid building such widely used models from scratch.
* Provide implementations of some advanced tools, including some special optimizers and loss functions.
* Currently, the pyTorch `DataLoader` does not support managing a large-file dataset in the initialization function. To manage the data more efficiently, we provide interfaces for loading large datasets like [HDF5 files][link-hdf5] by parallel. The alternative for transformers is also provided.
* Some APIs related to file IO and online requests are not safe enough. We wrap them by context and guarantee these ops are safe when errors occur.
* Provide some useful tools like record visualizers, and some open-sourced third-party tools.

## Current progress

Now we have such progress on the semi-product:

* [ ] `optimizers`
* [ ] `modules`
    * [x] `conv`: Modern convolutional layers and networks. [=100% "100%"]
    * [x] `resnet`: Residual blocks and networks. [=100% "100%"]
    * [ ] `resnext`: ResNeXt blocks and networks. [=0% "0%"]
    * [ ] `incept`: Google inception blocks and networks. [=0% "0%"]
    * [ ] `densenet`: Dense-net blocks and networks. [=0% "0%"]
* [ ] `models`
* [ ] `data`
    * [x] `h5py`: Wrapped HDF5 datasets saver and loader. [=100% "100%"]
    * [ ] `netcdf4`: Wrapped NETCDF4 datasets saver and loader. [=0% "0%"]
    * [ ] `bcolz`: Wrapped Bcolz datasets saver and loader. [=0% "0%"]
    * [ ] `text`: Wrapped text-based datasets saver and loader (CSV, JSON, TXT). [=0% "0%"]
    * [x] `preprocs`: Useful pre- and post- processing tools for all data handles in this package. [=100% "100%"]
    * [x] `webtools`: Web tools for downloading tarball-packed datasets from Github. [=100% "100%"]
* [ ] `funcs`
* [ ] `utils`
    * [ ] `tools`: Light-weighted recording parsing tools used during training or testing. [=10% "10%"]
    * [ ] `draw`: Wrapped `matplotlib` drawing tools. Most of the utilities are designed as call-back based functions. [=80% "80%"]
* [ ] `contribs`
    * [x] `torchsummary` [:fontawesome-solid-external-link-alt:](https://github.com/sksq96/pyTorch-summary): Keras style model.summary() in pyTorch, with some bugs gotten fixed (modified) (MIT licensed). [=100% "100%"]
    * [ ] `tensorboard` [:fontawesome-solid-external-link-alt:](https://pyTorch.org/docs/stable/tensorboard.html): Wrapped `torch.utils.tensorboard`, supporting context-style writer and tensorboard.log converted to `h5py` format (not modified). [=0% "0%"]

## Compatibility test

???+ info
    Currently, this project has not been checked by compatibility tests. During the developing stage, we are using pyTorch 1.7.0+ and Python 3.6+.

To perform the compatibility test, just run

```bash
python -m mdnc
```

The compatibility test is shown as below. The checked item means this package performs well in the specific enviroment.

| Enviroment | Win | Linux |
| :---- | :----: | :----: |
| pyTorch 1.7.0, Python 3.8 | :fontawesome-solid-check: | |
| pyTorch 1.8.0, Python 3.8 | | |
| pyTorch 1.6.0, Python 3.7 | | |
| pyTorch 1.4.0, Python 3.7 | | |
| pyTorch 1.2.0, Python 3.6 | | |
| pyTorch 1.0.0, Python 3.5 | | |

[link-hdf5]:https://www.hdfgroup.org/solutions/hdf5 "HDF5"