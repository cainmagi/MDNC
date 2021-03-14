# utils.tools.ContextWrapper

:codicons-symbol-class: Class · :codicons-symbol-field: Context · [:octicons-file-code-24: Source]({{ source.root }}/utils/tools.py#L75){ target="_blank" }

```python
inst_wctx = mdnc.utils.tools.ContextWrapper(
    instance, exit_method=None
)
```

A simple wrapper for adding context support to some special classes.

???+ example
    For example, there is an instance f, it defines f.close(), but does not support the context. In this case, we could use this wrapper to add context support:

    ```python linenums="1"
    import mdnc
    f = create_f(...)
    with mdnc.utils.tools.ContextWrapper(f) as fc:
        do some thing ...
    # When leaving the context, the f.close() method would be called
    # automatically.
    ```

???+ tip
    Actually, the standard lib has already provided a tool with the similar usage. See [:fontawesome-solid-external-link-alt: contextlib.closing][pydoc-contextclose] for viewing the details.

    In most cases, we recommend to use the solution from the `contextlib`. However, if an instance implements an exiting function not named `close()`, then we have to use this class.

## Arguments

**Requries**

| Argument {: .w-5rem} | Type {: .w-4rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `instance` | `#!py object` | An instance requring the context support. |
| `exit_method` | `#!py object` | A callable object (function), if not provided, would call the `#!py instance.close()` method during the exiting stage. If provided, would call `#!py exit_method(instance)` instead. |

## Operators

### :codicons-symbol-operator: `#!py __enter__`, `#!py __exit__`

```python
with mdnc.utils.tools.ContextWrapper(instance, exit_method=None) as inst_wctx:
    ...
```

Work with the context. When leaving the context for any reason (sucessfully running all codes or meeting any exceptions), the `exit_method` would be called.

## Examples

???+ example
    === "Codes"
        ```python linenums="1"
        import time
        import tqdm
        import mdnc

        num_iters = 100
        with mdnc.utils.tools.ContextWrapper(tqdm.tqdm(total=num_iters)) as tq:
            for i in range(num_iters):
                tq.update(1)
                time.sleep(0.001)
        ```

    === "Output"
        ```
        100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 62.98it/s]
        ```

[pydoc-contextclose]:https://docs.python.org/3/library/contextlib.html#contextlib.closing "contextlib.closing"
