# utils.tools.EpochMetrics

:codicons-symbol-class: Class · [:octicons-file-code-24: Source]({{ source.root }}/data/h5py.py#L1351)

```python
emdict = mdnc.utils.tools.EpochMetrics(reducer=np.mean)
```

A dictionary for storing metrics. The [`__setitem__`](#__setitem__) and [`__getitem__`](#__getitem__) operators are overloaded. This tool is used for calculating the statistics of epoch metrics easily. The following codes are equivalent:

???+ example
    === "With EpochMetrics"
        ```python linenums="1"
        emdict = EpochMetrics()
        for i in range(10):
            emdict['loss'] = i / 10
            emdict['metric2'] = - i / 10
        print(emdict['loss'], emdict['metric2'])
        ```

    === "Without EpochMetrics"
        ```python linenums="1"
        emdict = {'loss': list()}
        for i in range(10):
            emdict['loss'].append(i / 10)
            emdict['metric2'].append(- i / 10)
        print(np.mean(emdict['loss']), np.mean(emdict['metric2']))
        ```

## Arguments

**Requries**

| Argument {: .w-5rem} | Type {: .w-4rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `reducer` | `#!py object` | A callable object (function). The input of this function should be a sequence and the output should be a scalar. |

## Methods

### :codicons-symbol-method: `keys`, `values`, `items`

```python
for k in emdict.keys():
    ...
for v in emdict.values():
    ...
for k, v in emdict.items():
    ...
```

Used as iterators returned by a python dictionary.

-----

### :codicons-symbol-method: `pop`, `popitem`

```python
v = emdict.pop(k, default=None)
k, v = emdict.popitem(k)
```

Used as `pop()` and `popitem()` of a python dictionary.

-----

### :codicons-symbol-method: `get`, `setdefault`

```python
v = emdict.get(k, default=None)
emdict.setdefault(k, default=None)
```

Used as `get()` and `setdefault()` of a python dictionary.

## Operators

### :codicons-symbol-operator: `#!py __getitem__`

```python
v = emdict[keyword]
```

Get the reduced value of a specific keyword.

**Requries**

| Argument {: .w-5rem} | Type {: .w-4rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `keyword` | `#!py object` | A python object that could be used as the keyword. This is the name of the metric. |

**Returns**

| Argument {: .w-5rem} | Description {: .w-8rem} |
| :------: | :---------- |
| `v` | The returned metric, this value is a scalar reduced by the `reducer` provided in the initialization. |

### :codicons-symbol-operator: `#!py __setitem__`

```python
emdict[keyword] = v
```

Set a new value for a specific keyword.

**Requries**

| Argument {: .w-5rem} | Type {: .w-4rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `keyword` | `#!py object` | A python object that could be used as the keyword. This is the name of the metric. |
| `v` | `#!py int` or<br>`#!py float` | A scalar value. This value would be appended in the stored metric list. |

## Examples

???+ example
    === "Codes"
        ```python linenums="1"
        import mdnc

        emdict = mdnc.utils.tools.EpochMetrics()
        for i in range(10):
            emdict['loss'] = i / 10
            emdict['metric2'] = - i / 10
        for k, v in emdict.items():
            print(k, v)
        ```

    === "Output"
        ```
        loss 0.45
        metric2 -0.45
        ```
