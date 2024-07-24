RAF Model Zoo and Benchmarking
===============================

This repository provides a model zoo of RAF and a benchmark infra to benchmark its performance.

Requirements
------------

- RAF
- torch >= 1.8.1
- torchvision >= 0.8.2 (tested on 0.9.1).
- transformers >= 3.5 (tested on 4.3).

Usages
------

`benchmark` Python package has command-line interface (CLI) so that you can directly benchmark a certain model without writing a Python script.
Here are some examples. For more detail descriptions, please see `python3 -m benchmark -h`.

```
# Benchmark ResNet-50 performance with batch size 32 on RAF.
python3 -m benchmark bench --framework raf --model resnet50 --batch 32 --sch-file sch/sch_amp_off_best.json

# Benchmark ResNet-50 performance with batch size 32 on RAF with AMP enabled.
python3 -m benchmark bench --framework raf --model resnet50 --batch 32 --amp --sch-file sch/sch_amp_on_best.json

# Benchmark ResNet-50 performance on PyTorch.
python3 -m benchmark bench --framework torch --model resnet50 --batch 32

# Check correctness by comparing the loss values of ResNet-50 between RAF and PyTorch.
python3 -m benchmark check_correctness --framework raf --model resnet50 --batch 32

# Profile ResNet-50 performance.
# The result will be dumped to a log file with "profile_" as its prefix.
python3 -m benchmark profile_latency --framework raf  --model resnet50 --batch 32 --sch-file sch/sch_amp_off_best.json

# Profile ResNet-50 memory footprint.
python3 -m benchmark profile_memory --framework raf  --model resnet50 --batch 32

# Trace ResNet-50 memory over the execution.
# The result will be dumped to a log file with "trace_" as its prefix.
python3 -m benchmark trace_memory --framework raf --model resnet50 --batch 32

# Tune a set of ops in BERT.
python3 -m benchmark tune --framework raf --model bert_base_mlm --batch 32 --sch-file bert.json \
        --only-tune-tasks take_dx,softmax_dx,layer_norm,layer_norm_dx

# Benchmark ResNet-50 performacne with batch size 32 on PyTorch/XLA on GPU
GPU_NUM_DEVICES=1 python3 -m benchmark bench --framework torch --model resnet50 --batch 32 --ltc xla

# Benchmark ResNet-50 performacne with batch size 32 on RAZOR on GPU
ENABLE_PARAM_ALIASING=true RAZOR_DEVICE=GPU python3 -m benchmark bench --framework torch --model resnet50 --batch 32 --ltc razor
```

Meanwhile, `benchmark` also supports benchmarking on multiple devices on one instance.
Note that it requires MPI and NCCL.
See https://github.com/awslabs/raf/blob/main/docs/wiki/1_getting_start/Build-on-Ubuntu-18.04.md 
for NCCL installation, and make sure RAF_USE_MPI and RAF_USE_NCCL are set to ON in config.cmake.

For optimizers, `benchmark` uses `SGD` by default. It also supports LANS (https://arxiv.org/abs/2006.13484) for BERT and its variant by using flag `--optimizer LANS`. 

```
# Benchmark ResNet-50 performance with data parallel on 4 GPUs.
mpirun -np 4 python3 -m benchmark bench --framework raf --model resnet50 --batch 256 --data-parallel --sch-file sch/sch_amp_off_best.json

# Benchmark ResNet-50 performance with data parallel and ZeRO-1 on 4 GPUs.
mpirun -np 4 python3 -m benchmark bench --framework raf --model resnet50 --batch 256 --data-parallel --zero-opt 1 --sch-file sch/sch_amp_off_best.json
```

Here we illustrate the functionalities of the `benchmark` Python package.
Each illustraction includes a complete example that can directly be used as a script.

- List all available models and pull one of them. The returned `bencher` is a wrapper with some useful APIs to help benchmark the model.

    ```python
    import benchmark
    benchmark.list_model_names()
    bencher = benchmark.get_model_bencher("raf", "lenet5", batch_size=32)
    ```

    If you encounter the following error when pulling the model, it means the model failed to be converted from PyTorch.

    ```python
    >>> bencher = benchmark.get_model_bencher("raf", "bert", batch_size=32) # Get a model
    [2021-02-23 01:19:09] WARNING PyTorch-NLP: Failed to convert model to RAF: Traceback (most recent call last):
      # ... skip ...
    TVMError: One or more ops cannot be converted:
    Failed to convert nn.dropout (appear 37 times)
    ```

- Register a customized PyTorch model for benchmarking/testing:

    ```python
    import torch
    import raf
    from raf.frontend.pytorch import from_pytorch
    from raf.testing import randn_torch, one_hot_torch

    import benchmark
    from benchmark import reg_model, RAFBencher, TorchBencher

    @reg_model("raf")
    def mynet(batch_size, image_size, dtype, include_orig_model=False):
        class MyNet(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
                self.bn = torch.nn.BatchNorm2d(6)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn(x)  # (32, 6, 24, 24)
                x = x.view(x.shape[0], -1)
                return x

        t_model = MyNet()
        image_size = image_size if image_size is not None else (28, 28)
        input_shape = (batch_size, 3, *image_size)

        m_x, t_x = randn_torch(input_shape, dtype=dtype)
        t_model.eval()
        m_dy, t_dy = randn_torch((), std=0.0, mean=1.0, requires_grad=False)
        m_ytrue, t_ytrue = one_hot_torch(batch_size=batch_size, num_classes=6 * 24 * 24)

        ref_bencher = (
            TorchBencher(t_model, input_shape, [t_x], t_dy, t_ytrue) if include_orig_model else None
        )
        try:
            m_model = from_pytorch(t_model, {"input0": ((input_shape, dtype))})
        except Exception as err:
            print("Failed to convert model to RAF: %s", str(err))
            m_model = None
        return RAFBencher(m_model, input_shape, [m_x], m_dy, m_ytrue, ref_bencher=ref_bencher)

    bencher = benchmark.get_model_bencher("raf", "mynet", batch_size=32, include_orig_model=True)
    bencher.check_correctness(device="cuda", train=True)
    ```
