# data.webtools.download_tarball_link

:codicons-symbol-method: Function · [:octicons-file-code-24: Source]({{ source.root }}/data/webtools.py#L92){ target="_blank" }

```python
mdnc.data.webtools.download_tarball_link(
    link, path='.', mode='auto', verbose=False
)
```

Download an online tarball from a web link, and extract it automatically. This function is equivalent to using `wget`. For example, downloading a `xz` file:

```bash
wget -O- <link>.tar.xz | tar xJ -C <path>/ || fail
```

The tarball is directed by the link. The tarball would be sent to pipeline and not get stored.

Now supports `gz`, `bz2` or `xz` format, see [:fontawesome-solid-external-link-alt: tarfile][pydoc-tarfile] to view the details.

## Arguments

**Requries**

| Argument {: .w-5rem} | Type {: .w-4rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `link`  | `#!py str` | The whole web link, pointting to or redicted to the data file. |
| `path`  | `#!py str` | The extracted data root path. Should be a folder path. |
| `mode`  | `#!py str` | The mode of extraction. Could be `#!py 'gz'`, `#!py 'bz2'`, `#!py 'xz'` or `#!py 'auto'`. When using `#!py 'auto'`, the format would be guessed by the posfix of the file name in the link. |
| `verbose`  | `#!py bool` | A flag, whether to show the downloaded size during the web request. |

## Examples

???+ example
    === "Codes"
        ```python linenums="1"
        import mdnc

        mdnc.data.webtools.download_tarball_link('https://github.com/cainmagi/Dockerfiles/releases/download/xubuntu-v1.5-u20.04/share-pixmaps.tar.xz', path='./downloads', verbose=True)
        ```

    === "Output"
        ```
        Get share-pixmaps.tar.xz: 134kB [00:00, 1.65MB/s]
        ```

[pydoc-tarfile]:https://docs.python.org/3/library/tarfile.html "Read and write tar archive files"
