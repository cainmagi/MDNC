# data.webtools.download_tarball_private

:codicons-symbol-method: Function · [:octicons-file-code-24: Source]({{ source.root }}/contribs/torchsummary.py#L58)

```python
download_tarball_private(
    user, repo, tag, asset,
    path='.', mode='auto', token=None, verbose=False
)
```

Download an online tarball from a Github release asset, and extract it automatically (private).

This tool should only be used for downloading assets from private repositories. Although it could be also used for public repositories, we do not recommend to use it in those cases, because it would still require a token even the repository is public.

Now supports `gz`, `bz2` or `xz` format, see [:fontawesome-solid-external-link-alt: tarfile][pydoc-tarfile] to view the details.

???+ warning
    Using the general interface [`mdnc.data.webtools.download_tarball`](../download_tarball) is more recommended.

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
| `token` | `#!py str` | A given OAuth token. Only when this argument is unset, the program will try to find a token from enviornmental variables. To learn how to set the token, please refer to [`mdnc.data.webtools.get_token`](../get_token). |
| `verbose`  | `#!py bool` | A flag, whether to show the downloaded size during the web request. |

## Examples

???+ example
    === "Codes"
        ```python linenums="1"
        import mdnc

        mdnc.data.webtools.download_tarball_private(user='cainmagi', repo='React-builder-for-static-sites', tag='0.1', asset='test-datasets-1.tar.xz', path='./downloads', token='', verbose=True)
        ```

    === "Output"
        ```
        data.webtools: A Github OAuth token is required for downloading the data in private repository. Please provide your OAuth token:
        Token:****************************************
        data.webtools: Tips: specify the environment variable $GITTOKEN or $GITHUB_API_TOKEN could help you skip this step.
        Get test-datasets-1.tar.xz: 216B [00:00, 217kB/s]
        ```

[pydoc-tarfile]:https://docs.python.org/3/library/tarfile.html "Read and write tar archive files"
