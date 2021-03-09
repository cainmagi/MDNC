# data.webtools.download_tarball

:codicons-symbol-method: Function · [:octicons-file-code-24: Source]({{ source.root }}/contribs/torchsummary.py#L58)

```python
download_tarball(
    user, repo, tag, asset, path='.', mode='auto', token=None, verbose=False
)
```

Download an online tarball from a Github release asset, and extract it automatically.

This tool is used for downloading the assets from github repositories. It would:

1. Try to detect the data info in public mode;
2. If fails (the Github repository could not be accessed), switch to private downloading mode. The private mode requires a Github OAuth token for getting access to the file.
3. The tarball would be sent to pipeline and not get stored.

Now supports `gz`, `bz2` or `xz` format, see [:fontawesome-solid-external-link-alt: tarfile][pydoc-tarfile] to view the details.

???+ tip
    The mechanics of this function is a little bit complicated. It is mainly inspired by the following codes:
    
    * https://gist.github.com/devhero/8ae2229d9ea1a59003ced4587c9cb236
    * https://gist.github.com/maxim/6e15aa45ba010ab030c4

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
| `token` | `#!py str` | A given OAuth token. Only when this argument is unset, the program will try to find a token from enviornmental variables. To learn how to set the token, please refer to [`mdnc.data.webtools.get_token`](../get_token.md). |
| `verbose`  | `#!py bool` | A flag, whether to show the downloaded size during the web request. |

## Examples

???+ example "Example 1"
    === "Codes"
        ```python linenums="1"
        import mdnc

        mdnc.data.webtools.download_tarball(user='cainmagi', repo='Dockerfiles', tag='xubuntu-v1.5-u20.04', asset='xconfigs-u20-04.tar.xz', path='./downloads', verbose=True)
        ```

    === "Output"
        ```
        Get xconfigs-u20-04.tar.xz: 3.06kB [00:00, 263kB/s]
        ```

???+ example "Example 2"
    === "Codes"
        ```python linenums="1"
        import mdnc

        mdnc.data.webtools.download_tarball(user='cainmagi', repo='React-builder-for-static-sites', tag='0.1', asset='test-datasets-1.tar.xz', path='./downloads', token='', verbose=True)
        ```

    === "Output"
        ```
        data.webtools: A Github OAuth token is required for downloading the data in private repository. Please provide your OAuth token:
        Token:****************************************
        data.webtools: Tips: specify the environment variable $GITTOKEN or $GITHUB_API_TOKEN could help you skip this step.
        Get test-datasets-1.tar.xz: 216B [00:00, 217kB/s]
        ```

[pydoc-tarfile]:https://docs.python.org/3/library/tarfile.html "Read and write tar archive files"
