# data.webtools.get_token

:codicons-symbol-method: Function · [:octicons-file-code-24: Source]({{ source.root }}/contribs/torchsummary.py#L58)

```python
token = get_token(
    token='', silent=True
)
```

Automatically get the Github OAuth token, if the argument `token` is missing.

This function would try to get the token in the following orders:

1. Try to find the value of the environmental variable `GITTOKEN`.
2. If not found, try to find the value of the environmental variable `GITHUB_API_TOKEN`.
3. If not found, and `#!py silent is False`, would ask users to input the token. When `#!py silent is True`, would return `#!py ''`.

???+ tip
    How to get the token? Please read this page:

    [:fontawesome-solid-external-link-alt: Git automation with OAuth tokens][github-settoken]

???+ tip
    The token could be formatted like the following two forms:

    === "With user name"

        ```bash
        GITTOKEN=myusername:b05bpgw2dcn5okqpeltlz858eoi6x6j3wrrjhhhc
        ```

    === "Without name"

        ```bash
        GITTOKEN=b05bpgw2dcn5okqpeltlz858eoi6x6j3wrrjhhhc
        ```

    Another tip is that, you could skip entering the user name and password if you clone a private repository like this:
    
    ```bash
    git clone https://myusername:b05bpgw2dcn5okqpeltlz858eoi6x6j3wrrjhhhc@github.com/myusername/myreponame.git myrepo
    ```

    A repository cloned by this way does not require the user name and password for `pull` and `push`.

## Arguments

**Requries**

| Argument {: .w-5rem} | Type {: .w-4rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `token`  | `#!py str`  | The given OAuth token. Only when this argument is unset, the program will try to find a token from enviornmental variables. |
| `silent` | `#!py bool` | A flag. If set `#!py True` and the token could not be found anywhere, this tool would not ask for a token, but just return `#!py ''`. |

**Returns**

| Argument {: .w-5rem} | Description {: .w-8rem} |
| :------: | :---------- |
| `token` | A `#!py str`. This is the detected OAuth token. |

## Examples

???+ example
    === "Codes"
        Run `bash`,

        ```bash linenums="1"
        $GITTOKEN=xxxxxxxxxxxxxx
        ```

        Then, run `python`,

        ```python linenums="1"
        import mdnc

        token = mdnc.data.webtools.get_token(token='')
        print(token)
        ```

    === "Output"
        ```
        xxxxxxxxxxxxxx
        ```

[github-settoken]:https://docs.github.com/en/github/extending-github/git-automation-with-oauth-tokens "Git automation with OAuth tokens"
