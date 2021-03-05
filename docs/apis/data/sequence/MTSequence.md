# data.sequence.MTSequence

:codicons-symbol-class: Class · [:octicons-file-code-24: Source]({{ source.root }}/data/h5py.py#L1351)

```python
manager = mdnc.data.sequence.MTSequence(
    worker, dset_size, num_workers=4, num_converters=None, batch_size=32,
    buffer=10, shuffle=True, out_type='cuda', seed=None
)
```

This class is a scheduler based on multi-threading. It is designed as an alternative [`keras.utils.Sequence`][keras-sequence]. The multi-threading codes are built on top of the [`threading`][pydoc-threading] module. It supports different workers and allows users to read datasets asynchronously and shuffle dataset randomly.

This class could be loaded without pyTorch.

The workflow of this class is described in the following figure:

{% raw %}
```mermaid
flowchart LR
    subgraph indexer [Indexer]
        data[(Data)]
        getitem["__getitem__()"] --x data
    end
    mseq:::msequenceroot
    subgraph mseq [MTSequence]
        subgraph procs [Threading Pool]
            proc1[[Thread 1]]
            proc2[[Thread 2]]
            procn[[...]]
            subgraph indexer1 [Indexer1]
                getitem1["__getitem__()"]
            end
            subgraph indexer2 [Indexer2]
                getitem2["__getitem__()"]
            end
            subgraph indexern [...]
                getitemn["__getitem__()"]
            end
            proc1 -->|invoke| getitem1 --> data1[(Data 1)]
            proc2 -->|invoke| getitem2 --> data2[(Data 2)]
            procn -->|invoke| getitemn --> datan[(...)]
        end
        subgraph procs2 [Threading Pool 2]
            cvt1[[Type converter 1]] --> datam1[(Data 1)]
            cvtn[[...]] --> datamn[(...)]
        end
        data1 & data2 & datan -->|send| queue_m
        cvt1 & cvtn -->|fetch| queue_m
        datam1 & datamn -->|send| queue_o
        queue_i{{Input queue}}
        queue_m{{Middle queue}}
        queue_o{{Output queue}}
        mainthread["Main<br>thread"] -->|generate| indices[(Indices)]
        indices -->|send| queue_i
        mainthread -->|fetch| queue_o
    end
    proc1 & proc2 & procn -->|fetch| queue_i
    indexer -->|copy| indexer1 & indexer2 & indexern
    classDef msequenceroot fill:#FEEEF0, stroke: #b54051;
```
{% endraw %}

The workflow could be divided into steps:

1. An indexer is initialized outside of the `MTSequence`. The indexer would maintain the dataset during the initialization, and provide a `#! __getitem__(bidx)` method, where the argument `bidx` is a sequence of indicies. This method would read the dataset according to the indices and return a mini-batch of data in the `#! np.ndarray` format.
2. The `MTSequence` would store the indexer during the initialization.
3. When the [`start()`](#start) method is invoked, two threading pools would be created. The first pool maintains several threads, each thread would get a copy of the indexer provided in step 1. The second pool maintains several output data type converters. These converters are designed in MDNC and do not require users to implement.
4. There are 3 queues maintained by `MTSequence`. During the asynchronous data parsing, the main thread would generate a sequence of indicies in the beginning of each epoch. The indicies would be depatched to these parallel threads (in pool 1) by the **input queue**. Each thread would listen to the event of the input queue and try to get the depatched indicies. Once getting a sequence of indicies, the thread would invoke the `#! __getitem__()` method of its indexer, the output data would be sent to the second queue, i.e. the **middle queue**.
5. The converters in pool 2 would listen to the middle queue, get the mini-batches, and convert them to `#! torch.Tensor` or `#! torch.cuda.Tensor`. The converted data would be sent to the last queue, i.e. the **output queue**.
6. The main thread is an iterator. It keeps listening the output queue during the workflow. Once the `#!py __next__()` method is invoked, it would get one output mini-batch from the **output queue**. This behavior would repeat until the [`finish()`](#finish) method is invoked (or the context is closed).

## Arguments

**Requries**

| Argument {: .w-6rem} | Type {: .w-4rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `worker` | `#!py type` | A class used for generating worker instances, with `#!py __getitem__()` method implemented. This instance would be copied and used as indexer for different threads. |
| `dset_size` | `#!py int` | The number of samples in the dataset. If given an `#!py np.ndarray`, the array would be used as indices, the size of the dataset would be inferred as the length of the array. |
| `num_workers` | `#!py int` | The number of parallel workers, each worker is created by the argument `#!py worker()` inside the threads. |
| `num_converters` | `#!py int` | The number of converters, only used when cuda is enabled. If set `#!py None`, would be determined by `num_workers`. |
| `batch_size` | `#!py int` | The number of samples in each batch, used for depatching the indicies. |
| `shuffle` | `#!py bool` | If enabled, shuffle the dataset at the end of each epoch. |
| `out_type` | `#!py str` | The output type. Could be `#!py 'cuda'`, `#!py 'cpu'` or `#!py 'null'`. If set `#!py 'null'`, the results would not be converted to `torch.Tensor`. |
| `num_workers` | `#!py int` | The number of parallel workers. |
| `seed` | `#!py int` | : the seed used for shuffling the data. If not set, would use random shuffle without seed. |

??? info
    The argument `worker` does not require to be [picklable][pydoc-picklable] in this case, because all threads are mainted in the same process.

## Methods

### :codicons-symbol-method: `start`

```python
manager.start(compat=None)
```

Start the threading pool. When this method is invoked, the thereading pools would be initialized. It supports context management.

Running `start()` or `start_test()` would interrupt the started sequence.

**Requries**

| Argument {: .w-5rem} | Type {: .w-4rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `compat` | `#!py bool` | Only reserved for compatibility for switching from [`MPSequence`](../MPSequence) to `MTSequence`. This flag would not influence anything. |

??? tip
    This method supports context management. Using the context is recommended. Here we show two examples:
    === "Without context"
        ```python linenums="1"
        manager.start()
        for ... in manager:
            ...
        manager.finish()
        ```

    === "With context"
        ```python linenums="1"
        with manager.start() as mng:
            for ... in mng:
                ...
        ```

??? warning
    Even if you set `#!py shuffle=False`, due to the mechanism of the parallelization, the sample order during the iteration may still get a little bit shuffled. To ensure your sample order not changed, please use `#!py shuffle=False` during the initialization and use [`start_test()`](#start_test) instead.

-----

### :codicons-symbol-method: `start_test`

```python
manager.start_test(test_mode='default')
```

Start the test mode. In the test mode, the threading pool would not be open. All operations would be finished in the main thread. However, the random indices are still generated with the same seed of the parallel `#!py manager.start()` mode (if the indicies are not provided).

Running `start()` or `start_test()` would interrupt the started sequence.

**Requries**

| Argument {: .w-6rem} | Type {: .w-4rem} | Description {: .w-8rem} |
| :------: | :-----: | :---------- |
| `test_mode` | `#!py str` | Could be `#!py 'default'`, `#!py 'cpu'`, or `#!py 'numpy'`. <ul> <li>`#!py 'default'`: the output would be converted as `start()` mode.</li> <li>`#!py 'cpu'`: even set 'cuda' as output type, the testing output would be still not converted to GPU.</li> <li>`#!py 'numpy'`: would ignore all out_type configurations and return the original output. This output is still pre-processed.</li> </ul>  |

??? tip
    This method also supports context management. See [`start()`](#start) to check how to use it.

-----

### :codicons-symbol-method: `finish`

```python
manager.finish()
```

Finish the threading pool. The compatible mode would be auto detected by the previous `start()`.

## Properties

### :codicons-symbol-variable: `len()`, `length`

```python
len(dset)
manager.length
```

The length of the epoch. It is the number of mini-batches, also the number of iterations for each epoch.

-----

### :codicons-symbol-variable: `iter()`

```python
for x1, x2, ... in manager:
    ...
```

The iterator. Recommend to use it inside the context. The unpacked variables `#!py x1, x2 ...` are returned by the provided argument `worker`.

-----

### :codicons-symbol-variable: `dset_size`

```python
manager.dset_size
```

The size of the dataset. It contains the total number of samples for each epoch.

-----

### :codicons-symbol-variable: `batch_size`

```python
manager.batch_size
```

The size of each batch. This value is given by the argument `batch_size` during the initialization. The last size of the batch may be smaller than this value.

-----

### :codicons-symbol-variable: `use_cuda`

```python
manager.use_cuda
```

A `#!py bool`, whether to return `#! torch.cuda.Tensor`. This value would be only true when:

* The argument `out_type` is `#! 'cuda'`, or `#! 'cuda:x'` during the initialization.
* The pyTorch is available.

## Examples

???+ example "Example 1: default mode"
    === "Codes"
        ```python linenums="1"
        import mdnc

        class TestSequenceWorker:
            def __getitem__(self, indx):
                # print('data.sequence: thd =', indx)
                return indx

        manager = mdnc.data.sequence.MTSequence(TestSequenceWorker, dset_size=512, batch_size=1,
                                                out_type='cuda', shuffle=False, num_workers=1)
        with manager.start() as mng:
            for i in mng:
                print(i)
        ```

???+ example "Example 2: test mode"
    === "Codes"
        ```python linenums="1"
        import mdnc

        class TestSequenceWorker:
            def __getitem__(self, indx):
                # print('data.sequence: thd =', indx)
                return indx

        manager = mdnc.data.sequence.MTSequence(TestSequenceWorker, dset_size=512, batch_size=1,
                                                out_type='cuda', shuffle=False, num_workers=1)
        with manager.start_test('numpy') as mng:
            for i in mng:
                print(i)
        ```

[pydoc-picklable]:https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled "What can be pickled and unpickled?"
[pydoc-threading]:https://docs.python.org/3/library/threading.html "Thread-based parallelism"
[keras-sequence]:https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence "tf.keras.utils.Sequence"
