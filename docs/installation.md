# Installation

## Use the nightly version on Github

Currently, this project is still under-development. We suggest to use the following steps to add the package as a sub-module in your git-project,

```bash
cd <your-project-folder>
git submodule add https://github.com/cainmagi/MDNC.git mdnc
git submodule update --init --recursive
```

After that, you could use the pacakge by

```python
import mdnc.mdnc
```

If you want to update the sub-module to the newest version, please use

```bash
git submodule update --remote --recursive
```

## Install the package

???+ warning
    We strongly do not recommend to install the package by PyPI now. Because the pacakage is still under development.

This package could be also installed by the following command:

=== "Github"
    ```bash
    python -m pip install git+https://github.com/cainmagi/MDNC.git
    ```

=== "PyPI"
    ```bash
    to be implmented in the future...
    ```

Install the package by this way would make the package available globally. Make sure that the version is exactly what you want.
