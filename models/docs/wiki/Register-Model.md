Register a Model
================

In this document, we demonstrate how to register DenseNet-121 from torchvision for benchmarking.

1. Go to torchvision official [website](https://pytorch.org/vision/0.8/models.html) and find the model name: `densenet121`.
2. Go to `benchmark/pytorch/cv.py` and add the following code snippet. Note that the function name will also be the model name
   used in this benchmark.
    ```python
    @reg_model()
    def densenet121(batch_size, image_size, include_orig_model):
        return image_classify_common("densenet121", batch_size, image_size, include_orig_model)
    ```
3. Testing:
   ```
   >>> import benchmark
   >>> b = benchmark.get_model_bencher("densenet121")
   >>> b.bench("cuda", warmup=1)
   Cannot find config for target=cuda -keys=cuda,gpu -max_num_threads=1024 -thread_warp_size=32, workload=('dense_small_batch.cuda', ('TENSOR', (32, 1024), 'float32'), ('TENSOR', (1000, 1024), 'float32'), None, 'float32'). A fallback configuration is used, which may bring great performance regression.
   228.18750951904804
   >>> b.check_correctness("cuda")
   >>>
   ```

