"""
The module of wrapper classes to benchmark PyTorch models.
"""
# pylint: disable=too-many-arguments, too-many-statements
from importlib import import_module
import os
import timeit

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as torchDDP

from ..logger import get_logger
from ..model_bencher import ModelBencherBase
from ..dataset import transform_batch

logger = get_logger("Torch")  # pylint: disable=invalid-name

from .utils import (
    init_torch_ddp,
    init_deepspeed,
    init_horovod,
    to_torch_dev,
    turn_off_dropout,
    deepspeed_create_moe_param_groups,
    ddp_ignore_moe_parameters,
    ModelWithLoss,
)


class DeviceModelSetter:
    """Utilities of setting up device and model.

    Parameters
    ----------
    device : str
        The target device.
    """

    def __init__(self, device):
        self.device = torch.device(device)
        self.device_str = device

    def setup_model(self, model):
        """Setup the model."""
        model.to(self.device)
        model.train()
        return (model, model)

    def step(self):
        """A step after each iteration."""
        pass

    @property
    def autocast(self):
        """AMP context."""
        return torch.cuda.amp.autocast

    def optim(self, name, params):  # pylint: disable=no-self-use
        """Wrap the optimizer."""
        if name == "SGD":
            return torch.optim.SGD(params, lr=0.1, momentum=0.01)
        if name == "LANS":
            try:
                return import_module("apex.optimizers").FusedLANS(params)
            except ModuleNotFoundError:
                raise RuntimeError("apex.optimizers.FusedLANS is not imported corretly")
        raise ValueError(
            "only support SGD and LANS optimizers for now, but got " % name
        )

    @staticmethod
    def forward_amp_only():
        """Indicate whether the AMP should only cover the forward pass."""
        return True

    def sync(self):
        """perform device synchronization."""
        if self.device_str.startswith("cuda"):
            torch.cuda.synchronize(self.device)

    def finalize(self):
        """Finalize process."""
        self.sync()

    def profile_latency_start(self):
        """Start profiling latency."""
        raise NotImplementedError

    def profile_latency_stop(self):
        """Stop profiling latency."""
        raise NotImplementedError

    def get_latency_stats(self):
        """Get and reset the latency stats."""
        raise NotImplementedError

    def profile_memory_start(self):
        """Start profiling memory."""

    def profile_memory_stop(self):
        """Stop profiling memory."""

    def reset_memory_stats(self):
        """Reset memory stats."""
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def max_memory_allocated(self):
        """Get max memory allocated in MBs."""
        if self.device == "cuda":
            return torch.cuda.max_memory_allocated() / 2**20
        raise RuntimeError(f"Memory profiler on {self.device} is not supported")

    def max_memory_reserved(self):
        """Get max memory reserved in MBs."""
        if self.device == "cuda":
            return torch.cuda.max_memory_reserved() / 2**20
        raise RuntimeError(f"Memory profiler on {self.device} is not supported")

    @staticmethod
    def create(name, device):
        """The factory to dispatch the target setter."""
        if name == "xla":
            return TorchXLASetter()
        if name == "razor":
            return RAZORSetter()
        return DeviceModelSetter(device)


class TorchXLASetter(DeviceModelSetter):
    """The settings for LazyTensorCore with PyTorch/XLA. Note that we do not
    support mmeory profiling for PT-XLA yet.
    """

    def __init__(self):
        logger.info(
            "PyTorch/XLA is enabled. Device is controlled by env var GPU_NUM_DEVICES"
        )
        self.torch_xla = import_module("torch_xla")
        self.amp = import_module("torch_xla.amp")
        self.lm = import_module("torch_xla.core.xla_model")
        device = self.lm.xla_device()
        super().__init__(device)

    def setup_model(self, model):
        model = model.to(device=self.device)
        model.train()
        return (model, model)

    def step(self):
        self.lm.mark_step()

    @property
    def autocast(self):
        return self.amp.autocast

    @staticmethod
    def forward_amp_only():
        return True

    def finalize(self):
        """Print LTC metric report."""
        met = import_module("torch_xla.debug.metrics")
        print(met.metrics_report())

    def profile_latency_start(self):
        """Start profiling latency."""
        raise NotImplementedError

    def profile_latency_stop(self):
        """Stop profiling latency."""
        raise NotImplementedError

    def get_latency_stats(self):
        """Get and reset the latency stats."""
        raise NotImplementedError


class RAZORSetter(DeviceModelSetter):
    """The settings for LazyTensorCore with RAZOR."""

    def __init__(self):
        env = self.dump_env()

        self.razor = import_module("razor")
        self.lm = import_module("razor.lazy_tensor_core.core.lazy_model")
        self.raf = import_module("raf")
        self.razor_device = self.raf.Device(
            "cpu" if env["RAZOR_DEVICE"] == "CPU" else "cuda"
        )
        super().__init__("lazy")

    @staticmethod
    def dump_env():
        """Dump RAZOR environment information."""
        env_n_default = {"RAZOR_DEVICE": "CPU", "ENABLE_PARAM_ALIASING": "false"}

        env = {}
        logger.info("RAZOR Environment:")
        for env_name, default_value in env_n_default.items():
            val = os.environ.get(env_name, default_value)
            env[env_name] = val
            logger.info("\t%s: %s", env_name, val)
        return env

    def setup_model(self, model):
        model = model.to(device=self.device)
        model.train()
        return (model, self.razor.jit.script(model))

    def step(self):
        self.lm.mark_step()

    @property
    def autocast(self):
        return self.razor.amp.autocast

    # def optim(self, name, params):
    #     """Wrap the optimizer.
    #     FIXME(@comaniac): Somehow the RAZOR SGD results in OOM for some models.
    #     """
    #     if name == "SGD":
    #         return self.razor.optimizer.SGD(params, lr=0.1, momentum=0.01)
    #     if name == "LANS":
    #         return self.razor.optimizer.LANS(params)
    #     raise ValueError("only support SGD and LANS optimizers for now, but got " % name)

    @staticmethod
    def forward_amp_only():
        return False

    def finalize(self):
        """Print LTC metric report."""
        met = import_module("razor.lazy_tensor_core.debug.metrics")
        print(met.metrics_report(), flush=True)

    def profile_latency_start(self):
        """Start profiling latency."""
        self.raf.utils.profiler.get()  # Clear existing results if any.
        self.raf.utils.profiler.start()

    def profile_latency_stop(self):
        """Stop profiling latency."""
        self.raf.utils.profiler.stop()

    def get_latency_stats(self):
        """Get and reset the latency stats."""
        return self.raf.utils.profiler.get()

    def profile_memory_start(self):
        """Start profiling memory."""
        self.raf.utils.memory_profiler.reset()
        self.raf.utils.memory_profiler.start()

    def profile_memory_stop(self):
        """Stop profiling memory."""
        self.raf.utils.memory_profiler.stop()

    def reset_memory_stats(self):
        """Reset memory stats."""
        self.raf.utils.memory_profiler.reset()

    def max_memory_allocated(self):
        """Get max memory allocated in MBs."""
        mem_info = self.raf.utils.memory_profiler.get_max_memory_info(self.razor_device)
        assert (
            "max_used" in mem_info
        ), "Internal error: max_used is not found in memory info"
        return mem_info["max_used"].value

    def max_memory_reserved(self):
        """Get max memory reserved in MBs."""
        mem_info = self.raf.utils.memory_profiler.get_max_memory_info(self.razor_device)
        assert (
            "max_allocated" in mem_info
        ), "Internal error: max_allocated is not found in memory info"
        return mem_info["max_allocated"].value


class TorchBencher(ModelBencherBase):
    """The class to benchmark a PyTorch model.

    model: Any
        The model object from any framework.

    input_shape: Tuple[int, ...]
        The input shape of the model.

    args: List[Any]
        A list of input argument tensors.

    dy: Any
        A tensor of dy (only used in training).

    y_true: Any
        A tensor of y true (only used in training).
    """

    def __init__(self, model, input_shape, dataloader, dy, **kwargs):
        super().__init__(model, input_shape, dataloader, dy, **kwargs)
        if len(dy.shape) == 0:
            self.dy = self.dy.unsqueeze(0)
        self.executor = None
        self.md_setter = None
        self.data_parallel = False
        self.deepspeed = False
        self.horovod = False
        self.ds_config = None
        self.ds_model_engine = None
        self.batch_size = input_shape[0]
        self.output_shape = kwargs.get("reshape_output")
        assert self.output_shape is not None, "reshape_output must be set"

    def bench_infer_setup(self, device="cuda", **kwargs):
        self.args = [arg.to(device=device) for arg in self.args]
        self.model.to(device=device)

        # Since we convert PyTorch model to RAF for training,
        # the mathematic expression of PyTorch training model is the one
        # mapped to RAF model.
        self.model.train()

        def _run():
            with torch.no_grad():
                out = self.model(*self.args)
            if device == "cuda":
                torch.cuda.synchronize()
            return out

        self.executor = _run

    def bench_train_setup(self, device="cuda", **kwargs):
        device = to_torch_dev(device)
        self.deepspeed = kwargs.get("deepspeed", False)
        self.horovod = kwargs.get("horovod", False)
        assert not (
            self.deepspeed and self.horovod
        ), "Horovod cannot be used together with DeepSpeed."
        assert not (
            self.horovod and self.data_parallel
        ), "Horovod cannot be used together with Torch DDP."
        self.zero_opt = kwargs.get("zero_opt", False)
        # init distributed environments
        if self.deepspeed:
            ds_config = kwargs.get("ds_config_path", None)
            assert ds_config, "ds_config_path must be set when using deepspeed engine."
            self.ds_config = ds_config
            local_rank = init_deepspeed()
            device = "cuda:{}".format(local_rank)
            # need to wrap loss function here
            self.model = ModelWithLoss(
                self.model, self.kwargs.get("reshape_output", None)
            )
        else:
            self.data_parallel = kwargs.get("data_parallel", False)
            if self.data_parallel:
                _, _, local_rank = init_torch_ddp()
            elif self.horovod:
                _, _, local_rank = init_horovod()
            device = "cuda:{}".format(local_rank)

        self.md_setter = DeviceModelSetter.create(kwargs.get("ltc", None), device)
        device = self.md_setter.device

        amp = kwargs.get("amp", False)
        if amp:
            logger.info("AMP enabled in PyTorch.")

        _, self.model = self.md_setter.setup_model(self.model)

        is_checking_correctness = kwargs.get("check_correctness", False)
        if is_checking_correctness or self.deepspeed:
            logger.info("Turning off dropout layers.")
            turn_off_dropout(self.model)

        if self.deepspeed:
            import deepspeed

            is_moe = kwargs.get("pt_is_moe", False)

            parameters = filter(lambda p: p.requires_grad, self.model.parameters())
            if self.zero_opt and is_moe:
                parameters = deepspeed_create_moe_param_groups(self.model)

            self.ds_model_engine, optimizer, _, _ = deepspeed.initialize(
                model=self.model, model_parameters=parameters, config=self.ds_config
            )
        elif self.data_parallel:
            bucket_size = kwargs.get("bucket_size", None)
            moe_framework = kwargs.get("moe_framework", "")
            is_fastermoe = moe_framework == "fastermoe"
            if not is_fastermoe:
                ddp_ignore_moe_parameters(self.model)
                DDP = torchDDP
            else:
                from fmoe.distributed import DistributedGroupedDataParallel
                DDP = DistributedGroupedDataParallel
            if bucket_size is not None:
                self.model = DDP(
                    self.model, bucket_cap_mb=bucket_size, device_ids=[local_rank]
                )
            else:
                self.model = DDP(self.model, device_ids=[local_rank])

        if not self.deepspeed:
            optimizer_name = kwargs.get("optimizer", "SGD")
            optimizer = self.md_setter.optim(optimizer_name, self.model.parameters())
            if self.horovod:
                import horovod.torch as hvd

                optimizer = hvd.DistributedOptimizer(
                    optimizer,
                    named_parameters=self.model.named_parameters(),
                    compression=hvd.Compression.none,
                )

        # self.args = [arg.to(device=device) for arg in self.args]
        self.dy = self.dy.to(device=device)
        # self.y_true = self.y_true.to(device=device)

        # scalar = torch.cuda.amp.grad_scaler.GradScaler()
        def repeating_dataloader():
            while True:
                for batch in self.dataloader:
                    yield batch
        self.dataloader_generator = repeating_dataloader()

        def _run():
            batch = self.dataloader_generator.__next__()
            inputs, labels = transform_batch(batch)
            inputs = inputs.to(device=device)
            labels = labels.to(device=device)

            if self.deepspeed:
                loss = self.ds_model_engine(labels, inputs)
                self.ds_model_engine.backward(loss)
                self.ds_model_engine.step()
                return loss
            else:
                optimizer.zero_grad()

                def compute_loss():
                    # Wrap with a function to make sure the loss is the only
                    # alive tensor after the forward pass.
                    with self.md_setter.autocast(amp):
                        t_y = self.model(inputs)
                        if isinstance(t_y, tuple):
                            t_y = t_y[0]
                        elif isinstance(t_y, dict):
                            assert (
                                "logits" in t_y
                            ), "Expect ModelingOutputs with logits, but got %s" % type(
                                t_y
                            )
                            t_y = t_y["logits"]

                        # Reshape output if necessary.
                        reshape = self.kwargs.get("reshape_output", None)
                        t_y = t_y.view(*reshape) if reshape is not None else t_y

                        t_ypred = torch.log_softmax(t_y, dim=-1)
                        t_loss = torch.nn.functional.nll_loss(t_ypred, labels)
                        if isinstance(t_loss, tuple):
                            if hasattr(t_loss[0], "backward"):
                                t_loss = t_loss[0]
                            else:
                                assert hasattr(t_loss[1], "backward")
                                t_loss = t_loss[1]
                    return t_loss

                t_loss = compute_loss()

                with self.md_setter.autocast(
                    amp and not self.md_setter.forward_amp_only()
                ):
                    # Loss scaling with AMP.
                    # FIXME(comaniac): RAF does not have loos scaling yet, so turn off now
                    # to make a fair comparison.
                    if amp and False:
                        pass
                        # scalar.scale(t_loss).backward()
                        # scalar.step(optimizer)
                        # scalar.update()
                    else:
                        t_loss.backward()
                        optimizer.step()

                        if optimizer_name == "LANS":
                            optimizer.zero_grad()
                        else:
                            # Explicitly clear gradients to let them be optimized in LazyTensor IR.
                            if self.horovod:
                                optimizer.zero_grad()
                            else:
                                optimizer.zero_grad(set_to_none=True)

                    self.md_setter.step()
                print("Loss: {}".format(t_loss.item()), flush=True)
                return t_loss

        self.executor = _run

    def post_warmup(self):
        """Print the current metrics."""
        self.md_setter.finalize()

    def post_bench_timed(self):
        self.md_setter.sync()

    def post_bench(self):
        """Print the current metrics."""
        self.md_setter.finalize()
        if self.data_parallel:
            dist.destroy_process_group()
        if self.horovod:
            import horovod.torch as hvd

            hvd.shutdown()
        super().post_bench()

    def profile_latency(self, device="cuda", train=True, **kwargs):
        """Profile the latency of each execution kernel.

        Parameters
        ----------
        device: str
            The target device. Default "cuda".

        train: bool
            When set to True, evaluating the latency of SGD training. Default True.

        kwargs:
            See kwargs in bench_train_setup/bench_infer_setup for details.

        Returns
        -------
        Dict[str, Any]:
            The latency profiling result in JSON format.
        """
        if train:
            self.bench_train_setup(device=device, **kwargs)
        else:
            self.bench_infer_setup(device=device, **kwargs)

        # Warmup.
        timeit.repeat(self.executor, repeat=1, number=10)

        self.md_setter.profile_latency_start()
        timeit.repeat(self.executor, repeat=10, number=10)
        self.md_setter.profile_latency_stop()

        return self.md_setter.get_latency_stats()

    def profile_memory(self, show_used, device="cuda", train=True, **kwargs):
        """Profile the peak memory footprint.

        Parameters
        ----------
        show_used: bool
            Show the total used memory in MBs. If False, then it shows the total allocated
            memory.

        device: str
            The target device. Default "cuda".

        train: bool
            When set to True, evaluating the latency of SGD training. Default True.

        Returns
        -------
        float:
            The memory footprint in MBs.
        """
        assert device == "cuda", "Only support CUDA memory profiling for now"
        if train:
            self.bench_train_setup(device=device, **kwargs)
        else:
            self.bench_infer_setup(device=device, **kwargs)

        self.md_setter.reset_memory_stats()
        self.md_setter.profile_memory_start()
        timeit.repeat(self.executor, repeat=1, number=10)
        timeit.repeat(self.executor, repeat=10, number=10)
        self.md_setter.profile_memory_stop()
        memory_in_mbs = (
            self.md_setter.max_memory_allocated()
            if show_used
            else self.md_setter.max_memory_reserved()
        )
        self.md_setter.reset_memory_stats()
        return memory_in_mbs
