# data.webtools.download_tarball_public

:codicons-symbol-method: Function · [:octicons-file-code-24: Source]({{ source.root }}/contribs/torchsummary.py#L58)

```python
download_tarball_public(
    user, repo, tag, asset, path='.', mode='auto', verbose=False
)
```

Download an online tarball from a Github release asset, and extract it automatically (public).

This tool only supports public github repositories. This method could be replaced by [`mdnc.data.webtools.download_tarball_link`](../download_tarball_link), but we do not recommend to do that. Compared to that method, this function is more robust, because it fetches the meta-data before downloading the dataset.

Now supports `gz`, `bz2` or `xz` format, see [:fontawesome-solid-external-link-alt: tarfile][pydoc-tarfile] to view the details.

???+ warning
    This function is only designed for downloading public data. If your repository is private, please use [`mdnc.data.webtools.download_tarball_private`](../download_tarball_private) for instead. Certainly, using the general interface [`mdnc.data.webtools.download_tarball`](../download_tarball) is more recommended.

## Arguments

**Requries**

| Argument {: .w-5rem} | Type {: .w-4rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `user`  | `#!py str` | The Github owner name of the repository, could be an organization. |
| `repo`  | `#!py str` | The Github repository name. |
| `tag`   | `#!py str` | The Github release tag where the data is uploaded. |
| `asset` | `#!py str` | The github asset (tarball) name (including the file name postfix) to be downloaded. |
| `path`  | `#!py str` | The extracted data root path. Should be a folder path. |
| `mode`  | `#!py str` | The mode of extraction. Could be `#!py 'gz'`, `#!py 'bz2'`, `#!py 'xz'` or `#!py 'auto'`. When using `#!py 'auto'`, the format would be guessed by the posfix of the file name in the link. |
| `verbose`  | `#!py bool` | A flag, whether to show the downloaded size during the web request. |

## Examples

???+ example
    === "Codes"
        ```python linenums="1"
        import mdnc

        mdnc.data.webtools.download_tarball_public(user='cainmagi', repo='Dockerfiles', tag='xubuntu-v1.5-u20.04', asset='xconfigs-u20-04.tar.xz', path='./downloads', verbose=True)
        ```

    === "Output"
        ```
        Get xconfigs-u20-04.tar.xz: 3.06kB [00:00, 263kB/s]
        ```

[pydoc-tarfile]:https://docs.python.org/3/library/tarfile.html "Read and write tar archive files"
