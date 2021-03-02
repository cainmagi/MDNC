# Modern Deep Network Toolkits for pyTorch (MDNC)

This is a pyTorch framework used for

* Creating specially designed networks or layers.
* Parallel Data pre- and post- processing as a powerful alternative of `torch.utils.data.DataLoader`.
* Callback-function based data visualizer as an alternative of `seaborn`.
* Web tools for downloading tarball-packed datasets from Github.
* Some modified third-party utilities.

## Usage

Currently, this project is still under-development. We suggest to use the following steps to add the package as a sub-module in your git-project,

```bash
cd <your-project-folder>
git submodule add https://github.com/cainmagi/MDNC.git mdnc
git submodule update --init --recursive
```

After that, you could use the pacakge by

```python
from mdnc import mdnc
```

If you want to update the sub-module to the newest version, please use

```bash
git submodule update --remote --recursive
```

## Progress

Now we have such progress on the semi-product:

* [ ] `optimizers`
* [ ] `modules`
  * [x] `conv`: Modern convolutional layers and networks. ![100%](https://progress-bar.dev/100)
  * [x] `resnet`: Residual blocks and networks. ![100%](https://progress-bar.dev/100)
  * [ ] `resnext`: ResNeXt blocks and networks. ![0%](https://progress-bar.dev/0)
  * [ ] `incept`: Google inception blocks and networks. ![0%](https://progress-bar.dev/0)
  * [ ] `densenet`: Dense-net blocks and networks. ![0%](https://progress-bar.dev/0)
* [ ] `models`
* [ ] `data`
  * [x] `h5py`: Wrapped HDF5 datasets saver and loader. ![100%](https://progress-bar.dev/100)
  * [ ] `netcdf4`: Wrapped NETCDF4 datasets saver and loader. ![0%](https://progress-bar.dev/0)
  * [ ] `bcolz`: Wrapped Bcolz datasets saver and loader. ![0%](https://progress-bar.dev/0)
  * [ ] `text`: Wrapped text-based datasets saver and loader (CSV, JSON, TXT). ![0%](https://progress-bar.dev/0)
  * [x] `preprocs`: Useful pre- and post- processing tools for all data handles in this package. ![100%](https://progress-bar.dev/100)
  * [x] `webtools`: Web tools for downloading tarball-packed datasets from Github. ![100%](https://progress-bar.dev/100)
* [ ] `funcs`
* [ ] `utils`
  * [ ] `tools`: Light-weighted recording parsing tools used during training or testing. ![10%](https://progress-bar.dev/10)
  * [ ] `draw`: Wrapped `matplotlib` drawing tools. Most of the utilities are designed as call-back based functions. ![80%](https://progress-bar.dev/80)
* [ ] `contribs`
  * [x] `torchsummary` [:link:](https://github.com/sksq96/pyTorch-summary): Keras style model.summary() in pyTorch, with some bugs gotten fixed (modified) (MIT licensed). ![100%](https://progress-bar.dev/100)
  * [ ] `tensorboard` [:link:](https://pyTorch.org/docs/stable/tensorboard.html): Wrapped `torch.utils.tensorboard`, supporting context-style writer and tensorboard.log converted to `h5py` format (not modified). ![0%](https://progress-bar.dev/0)

## Documentation

To be built now...

## Demos

To be built now...

## Debug reports

Currently, this project has not been checked by compatibility tests. During the developing stage, we are using pyTorch 1.7.0+ and Python 3.6+.

To perform the compatibility test, just run

```bash
cd <root-of-this-repo>
python -m mdnc
```

The compatibility test is shown as below. The checked item means this package performs well in the specific enviroment.

| Enviroment | Win | Linux |
| :---- | :----: | :----: |
| pyTorch 1.7.0, Python 3.8 | :white_check_mark: | |
| pyTorch 1.8.0, Python 3.8 | | |
| pyTorch 1.6.0, Python 3.7 | | |
| pyTorch 1.4.0, Python 3.7 | | |
| pyTorch 1.2.0, Python 3.6 | | |
| pyTorch 1.0.0, Python 3.5 | | |

## Update reports

### 0.1.5 @ 3/2/2021

1. Add `DecoderNet` to our standard `module` protocol.
2. Fix some bugs of `data.h5py` and `data.preprocs`.

### 0.1.2 @ 2/27/2021

1. Fix more feature problems in `contribs.torchsummary`.
2. Fix bugs and finish `data.preprocs`.
3. Add more features in `data.webtools`.

### 0.1.0 @ 2/26/2021

1. Create this project.
2. Add packages: `contribs`, `data`, `modules`, `utils`.
3. Finish `modules.conv`, `modules.resnet`.
4. Finish `data.h5py`, `data.webtools`.
5. Finish `contribs.torchsummary`.
6. Drop the plan for supporting `contribs.tqdm`, add `utils.ContexWrapper` as for instead.
7. Add testing function for `data.webtools.DataChecker`.
