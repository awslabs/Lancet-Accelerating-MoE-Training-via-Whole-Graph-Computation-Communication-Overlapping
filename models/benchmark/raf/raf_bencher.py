"""
The module of wrapper classes to benchmark RAF models.
"""
# pylint: disable=too-many-arguments, unused-argument, protected-access,no-member,not-callable
from pathlib import Path
import timeit
import re
import os

import tvm
import raf
import numpy as np
from raf._core.executor import VMExecutor
from raf._core.device import Device
from raf.model.trace import _get_func_inputs
from raf.testing import check, numpy
from raf.utils.tuner import run_tuning
from raf import distributed as dist

from ..logger import get_logger
from ..model_bencher import ModelBencherBase
from ..dataset import transform_batch

from raf.utils.visualizer import draw_dataflow_graph
from raf._ffi.pass_ import GradientInputSelection, InlineLet, DeadCodeElimination, AutoCast, \
                           ToGraphNormalForm, ToBasicBlockNormalForm, FuseDialect, FuseTVM, \
                           DispatchDialect, EraseType, InferType, RevInlineAppliedBackward
from raf.ir.pass_manager import RAFSequential

logger = get_logger("RAF")  # pylint: disable=invalid-name


def torch_tensor_to_raf(inputs, labels, device):
    """Convert PyTorch tensor inputs and labels to RAF."""
    inputs = inputs.numpy()
    targets = labels.numpy()

    inputs = raf.array(inputs, device=device)
    labels = raf.array(targets, device=device)

    return inputs, labels

class RAFBencher(ModelBencherBase):
    """The class to benchmark a RAF model.

    model: raf.ModelBase
        The model object of RAF.

    input_shape: Tuple[int, ...]
        The input shape of the model.

    args: List[raf.ndarray]
        A list of input arguments.

    dy: Any
        A tensor of dy (only used in training).

    y_true: Any
        A tensor of y true (only used in training).

    ref_bencher: Optional[ModelBencherBase]
        Model bencher of the reference model.
    """

    def __init__(self, raf_model, input_shape, dataloader, dy, ref_bencher=None, **kwargs):
        super().__init__(raf_model, input_shape, dataloader, dy, **kwargs)
        self.ref_bencher = ref_bencher
        reshape = self.kwargs.get("reshape_output", None)
        assert reshape is not None, "reshape_output must be specified"
        self.output_shape = reshape

        self.model_w_loss = None
        if self.model is not None:
            if hasattr(self.model, "record"):
                inputs, labels = self._get_sample_data_from_dataloader()
                out = self.model.record(inputs)

                # Reshape output if necessary.
                out = raf._op.sym.reshape(out, reshape) if reshape is not None else out
                y_pred = raf._op.sym.log_softmax(out)
                if kwargs.get("dtype") == "float16":
                    # Cast to float32 for loss function.
                    y_pred = raf._op.sym.cast(y_pred, "float32")
                loss = raf._op.sym.nll_loss(labels, y_pred)
                self.model_w_loss = self.model + loss
            else:
                logger.info("Skip appending loss function because RAF model is not FrameworkModel")
                self.model_w_loss = self.model

        self.tvm_device = None
        self.vm_inputs = None
        self.vm = None
        self.executor = None
        self.trainer = None
        self.data_parallel = None
        self.zero_opt = None
        self.device = None
        self.fw_results = None

    def _get_sample_data_from_dataloader(self):
        """Get one batch of data from dataloader."""
        batch = self.dataloader.__iter__().__next__()
        inputs, labels = transform_batch(batch)
        inputs, labels = torch_tensor_to_raf(inputs, labels, "cpu")
        return inputs, labels

    def visualize_dag(self, output_file_name, device="cuda", train=True, draw_atomic_nodes=False, dump_as_text=False, **kwargs):
        if train:
            self.model_w_loss.train_mode()
            data_parallel = kwargs.get("data_parallel", False)
            overlap_comm_forward = kwargs.get("overlap_comm_forward", False)
            if data_parallel:
                dctx = dist.get_context()
                dctx.enable_data_parallel = True
                dctx.overlap_comm_forward = overlap_comm_forward
                assert raf.build.with_distributed(), " RAF is not build with distributed support"
                assert device == "cuda", "data parallel should run with device of cuda"
                device = "cuda({})".format(dctx.local_rank)
                device_type, device_id = device.split("(")
                device_id = int(device_id.strip()[:-1])
                self.tvm_device = tvm.nd.device(device_type, device_id)
            else:
                self.tvm_device = tvm.nd.device(device)
            self.args = [arg.to(device=device) for arg in self.args]
            for arg in self.args:
                arg.requires_grad = True
            if isinstance(self.dy, list):
                self.dy = tuple([y.to(device=device) for y in self.dy])
            else:
                self.dy = self.dy.to(device=device)
            self.model_w_loss.to(device=device)

            fuse_level = kwargs.get("fuse_level", 0)
            amp = kwargs.get("amp", False)
            trainer = raf.optim.sgd.with_sgd(learning_rate=0.1, momentum=0.01)(self.model_w_loss)[0]

            self._visualize_dag_impl(
                trainer,
                [self.dy, *self.args, self.y_true],
                output_file_name,
                fuse_level,
                draw_atomic_nodes,
                amp,
                dump_as_text
            )
        else:
            self.model.infer_mode()
            self.tvm_device = tvm.nd.device(device)
            self.args = [arg.to(device=device) for arg in self.args]
            self.model.to(device=device)

            fuse_level = kwargs.get("fuse_level", 0)
            amp = kwargs.get("amp", False)
            self._visualize_dag_impl(
                self.model,
                self.args,
                output_file_name,
                fuse_level,
                draw_atomic_nodes,
                amp,
                dump_as_text
            )

    def _visualize_dag_impl(
        self,
        model,
        args,
        out_file_name,
        fuse_level=0,
        draw_atomic_nodes=False,
        amp=False,
        dump_as_text=False):
        record = model._internal(*args)
        mod = record.mod
        config = {"raf.fuse_level": fuse_level}
        if amp:
            print("AMP enabled in RAF.")
            mod = raf._ffi.pass_.AutoCast()(mod)
            config["raf.amp.out_dtype"] = "float16"
        with tvm.transform.PassContext(
            opt_level=3,
            config=config,
            disabled_pass={"AutoSchedulerLayoutRewrite"},
        ):
            pass_seqs = [InferType(), GradientInputSelection(), InlineLet(), DeadCodeElimination(),
                        ToGraphNormalForm(), ToBasicBlockNormalForm(), InferType(), FuseDialect(),
                        FuseTVM(), DispatchDialect(), EraseType(), InferType()]
            mod = RAFSequential(pass_seqs)(mod)
        draw_dataflow_graph(mod, out_file_name=out_file_name, draw_atomic_nodes=draw_atomic_nodes, dump_as_text=dump_as_text)
        return

    @staticmethod
    def validate_sch_file(sch_file):
        """Validate the given schedule file and print a warning if the schedule file path
        is given but the file does not exist.

        Parameters
        ----------
        sch_file: Union[str, None]
            The path of schedule file or None.
        """
        if sch_file is not None:
            sch_file_path = Path(sch_file)
            if not sch_file_path.exists():
                logger.warning("Schedule file not found: %s", sch_file_path.absolute())

    def get_vm_exec(
        self,
        model,
        device,
        disable_fuse=False,
        sch_file=None,
        dryrun=False,
        amp=False,
        use_profile=False,
        disable_load_module=False,
        enable_lancet=False,
        partition_comm = False,
    ):
        """Helper function to initialize a VM to save to self.vm_inpus and self.vm_exec.

        Parameters
        ----------
        model: raf.ModelBase
            The model object of RAF.

        args: List[raf.ndarray]
            A list of input arguments.

        device: str
            The target device. Default "cuda".

        disable_fuse: bool
            Whether to disable fusion.

        sch_file: str
            The log file of tuning records.

        dryrun: bool
            Whether to dryrun (for tuning).

        amp: bool
            Whether to use AMP.

        Returns
        -------
        exec: Callable
            A function to execute the model using VM.
        """
        # first get a sample of the input and generate a trace record
        inputs, labels = self._get_sample_data_from_dataloader()
        record = model._internal(self.dy, inputs, labels)
        mod = record.mod
        if disable_load_module:
            use_profile = False
            enable_lancet = False
        config = {"raf.dp_schedule.use_profile": use_profile,
                "raf.dp_schedule.enable_lancet": enable_lancet,
                "raf.dp_schedule.disable_load_module": disable_load_module,
                "raf.dp_schedule.partition_large_comm": partition_comm}
        disabled_pass = ["AutoSchedulerLayoutRewrite"]
        if disable_fuse:
            disabled_pass += ["FuseDialect", "FuseTVM"]
        if dist.get_context().overlap_comm_forward:
            logger.info("Using RevInlineAppliedBackward.")
            mod = InferType()(mod)
            mod = RevInlineAppliedBackward()(mod)
            mod = InferType()(mod)
            logger.info("Finished applying RevInlineAppliedBackward.")
        if amp:
            logger.info("AMP enabled in RAF.")
            mod = AutoCast()(mod)
            config["raf.amp.out_dtype"] = "float16"
        # self.vm_inputs = _get_func_inputs(record, args, {}, get_handle=False)
        # if len(self.vm_inputs) != len(mod["main"].params) and self.fw_results is not None:
        #     self.vm_inputs += self.fw_results
        # assert len(self.vm_inputs) == len(mod["main"].params)

        with Device(device):
            with tvm.transform.PassContext(
                opt_level=3,
                config=config,
                disabled_pass=disabled_pass,
            ):
                self.vm = VMExecutor(mod, device, dryrun=dryrun)
                vm_exec = self.vm.make_executor(sch_file)

                def repeating_dataloader():
                    while True:
                        for batch in self.dataloader:
                            yield batch
                self.dataloader_generator = repeating_dataloader()
                self.iteration = 0

                def _run_vm_with_args():
                    self.iteration += 1
                    batch = self.dataloader_generator.__next__()
                    inputs, labels = transform_batch(batch)
                    inputs, labels = torch_tensor_to_raf(inputs, labels, device=device)
                    vm_inputs = _get_func_inputs(record, [self.dy, inputs, labels], {}, get_handle=False)
                    return vm_exec(*vm_inputs)
                return _run_vm_with_args

    def bench_infer_setup(self, device="cuda", **kwargs):
        """Setup the model to benchmark forward inference.

        Parameters
        ----------
        device: str
            The target device. Default "cuda".

        kwargs:
            use_interpreter: bool
                Use interpreter instead of VM to benchmerk.
            disable_fuse: bool
                Whether to disable fusion.
            sch_file: str
                The tuned log file path.
            dryrun: bool
                Whether to dryrun (for tuning).
        """
        self.model.infer_mode()
        self.data_parallel = kwargs.get("data_parallel", False)
        self.zero_opt = kwargs.get("zero_opt", 0)
        self.device = device
        assert (
            not self.data_parallel and self.zero_opt == 0
        ), "Doesn't support data parallel or ZeRO in infer mode"
        self.tvm_device = tvm.nd.device(device)
        self.args = [arg.to(device=device) for arg in self.args]
        self.model.to(device=device)

        dryrun = kwargs.get("dryrun", False)
        disable_fuse = kwargs.get("disable_fuse", False)
        amp = kwargs.get("amp", False)

        if "use_interpreter" in kwargs and kwargs["use_interpreter"]:
            assert not dryrun, "Dryrun is only available on VM"

            def run_interpreter():
                out = self.model(*self.args)
                self.tvm_device.sync()
                return out

            self.executor = run_interpreter
        else:
            sch_file = kwargs.get("sch_file", None)
            self.validate_sch_file(sch_file)
            vm_exec = self.get_vm_exec(
                self.model,
                self.args,
                device,
                disable_fuse,
                sch_file,
                dryrun,
                amp,
            )

            def _run_vm():
                out = vm_exec()
                self.tvm_device.sync()
                if isinstance(out, (raf._core.value.TupleValue, tuple, list)):
                    return out[0]
                return out

            self.executor = _run_vm

    def _check_value_in_args(self, tensor_value):
        # if not in expected_memory_type, then it must be one of
        # reshaped model inputs
        in_args = False
        data_ptr = tensor_value.data
        for arg in self.args:
            if arg._ndarray__value.data == data_ptr:
                in_args = True
                break
        assert in_args, "fw_outs must be either model fw outputs or reshaped model inputs"

    def create_or_update_fw_outputs(self, fw_outs):
        # check all fw_outs are using not owned async memory
        expected_memory_type = set(["NonOwnedAsyncMemory", "NonOwnedReservedAsyncMemory"])
        for t in fw_outs:
            if isinstance(t, (raf._core.value.TupleValue, tvm.ir.container.Array)):
                for idx in range(len(t)):
                    if (t[idx].memory_type() not in expected_memory_type):
                        self._check_value_in_args(t[idx])
            else:
                if (t.memory_type() not in expected_memory_type):
                    self._check_value_in_args(t)

        if self.fw_results is None:
            self.fw_results = []
            for t in fw_outs:
                if isinstance(t, (raf._core.value.TupleValue, tvm.ir.container.Array)):
                    result = []
                    for idx in range(len(t)):
                        result.append(raf.array(()))
                    self.fw_results.append(result)
                else:
                    self.fw_results.append(raf.array(()))

        assert len(fw_outs) == len(self.fw_results)
        for idx, t in enumerate(fw_outs):
            if isinstance(t, (raf._core.value.TupleValue, tvm.ir.container.Array)):
                # assume we dont have nested tuples
                assert isinstance(self.fw_results[idx], (tuple, list))
                assert len(self.fw_results[idx]) == len(t)
                for tuple_idx in range(len(t)):
                    self.fw_results[idx][tuple_idx].update_value(t[tuple_idx])
            else:
                self.fw_results[idx].update_value(t)
        # check all supplementary outputs are using not owned async memory
        for t in self.fw_results:
            if isinstance(t, (tuple, list)):
                # assume we dont have nested tuples
                for tuple_idx in range(len(t)):
                    if(t[tuple_idx]._ndarray__value.memory_type() not in expected_memory_type):
                        self._check_value_in_args(t[tuple_idx]._ndarray__value)
            else:
                if(t._ndarray__value.memory_type() not in expected_memory_type):
                    self._check_value_in_args(t._ndarray__value)

    def bench_train_setup(self, device="cuda", **kwargs):
        """Setup the model to benchmark backward training.

        Parameters
        ----------
        device: str
            The target device. Default "cuda".

        kwargs:
            use_interpreter: bool
                Use interpreter instead of VM to benchmerk.
            disable_fuse: bool
                Whether to disable fusion.
            amp: bool
                Whether use automatic mixed precision (AMP).
            sch_file: str
                The tuned log file path.
            dryrun: bool
                Whether to dryrun (for tuning).
            data_parallel: bool
                Whether to enable data (activation) parallel.
            zero_opt: int
                The ZeRO optimization level.
            optimizer: str
                The optimizer. option: SGD and LANS
        """
        def run_setup(use_profile=False, enable_lancet=False):
            nonlocal device
            self.model_w_loss.train_mode()

            # Process kwargs.
            self.data_parallel = kwargs.get("data_parallel", False)
            self.overlap_comm_forward = kwargs.get("overlap_comm_forward", False)
            partition_comm = kwargs.get("partition_comm", False)
            self.zero_opt = kwargs.get("zero_opt", 0)
            is_tuning = kwargs.get("is_tuning", False)
            dryrun = kwargs.get("dryrun", False)
            disable_fuse = kwargs.get("disable_fuse", False)
            amp = kwargs.get("amp", False)
            optimizer = kwargs.get("optimizer", "SGD")

            if self.data_parallel or self.zero_opt > 0:
                if not device.startswith("cuda"):
                    raise RuntimeError("Only support data parallel or ZeRO on CUDA")
                if not raf.build.with_distributed():
                    raise RuntimeError("RAF is not built with RAF_USE_MPI=ON and RAF_USE_NCCL=ON")

                dctx = dist.get_context()
                dctx.enable_data_parallel = self.data_parallel
                dctx.overlap_comm_forward = self.overlap_comm_forward
                dctx.zero_opt_level = self.zero_opt
                local_rank = dctx.local_rank
                if not is_tuning:
                    device = f"cuda({local_rank})"
            self.device = device

            if self.device.find("(") != -1:
                device_type = self.device.split("(")[0]
                device_id = int(self.device.split("(")[-1].split(")")[0])
                self.tvm_device = tvm.nd.device(device_type, device_id)
            else:
                self.tvm_device = tvm.nd.device(self.device)

            # self.args = [arg.to(device=self.device) for arg in self.args]
            # for arg in self.args:
            #     arg.requires_grad = True
            if isinstance(self.dy, list):
                self.dy = tuple([y.to(device=self.device) for y in self.dy])
            else:
                self.dy = self.dy.to(device=self.device)
            # self.y_true = self.y_true.to(device=self.device)
            self.model_w_loss.to(device=self.device)

            if "use_interpreter" in kwargs and kwargs["use_interpreter"]:
                assert not dryrun, "Dryrun is only available on VM"

                def run_interpreter():
                    loss = self.model_w_loss(*self.args, self.y_true)
                    loss.backward()
                    self.tvm_device.sync()
                    return loss

                self.executor = run_interpreter
            else:
                if optimizer == "SGD":
                    self.trainer, self.fw_only_trainer = raf.optim.sgd.with_sgd(learning_rate=0.1, momentum=0.01)(
                        self.model_w_loss
                    )
                elif optimizer == "LANS":
                    self.trainer, self.fw_only_trainer = raf.optim.lans.with_lans()(self.model_w_loss)
                else:
                    assert False, "only support SGD and LANS for now"
                sch_file = kwargs.get("sch_file", None)
                self.validate_sch_file(sch_file)

                vm_exec = self.get_vm_exec(
                    self.trainer,
                    self.device,
                    disable_fuse,
                    sch_file,
                    dryrun,
                    amp,
                    use_profile=use_profile,
                    enable_lancet = enable_lancet,
                    partition_comm = partition_comm
                )

                self.vm_exec = vm_exec
                # self.iteration_debug = -1
                # device_id = int(self.device.split("(")[-1].split(")")[0])

                # # DEBUG dump gradients
                # m_model = self.model_w_loss

                # assert m_model is not None
                # # Get all parameters from RAF model.
                # m_params = [p for p in dir(m_model) if p.startswith("model_")]
                # out_dir = f"./grads/device_{device_id}/iter_{self.iteration_debug}"
                # if not os.path.exists(out_dir):
                #     os.makedirs(out_dir)
                # for m_param in m_params:
                #     # Get the corresponding parameter in PyTorch model.
                #     attr = ""

                #     for token in m_param[6:].split("_"):
                #         if attr:
                #             attr += "_"
                #         attr += token
                #     m_param_tensor = getattr(m_model, m_param)
                #     np.save(os.path.join(out_dir, f"{attr}.npy"), m_param_tensor.numpy())
                # m_params = [p for p in dir(self.trainer) if p.startswith("model_")]
                # for m_param in m_params:
                #     # Get the corresponding parameter in PyTorch model.
                #     attr = ""

                #     for token in m_param[6:].split("_"):
                #         if attr:
                #             attr += "_"
                #         attr += token
                #     m_param_tensor = getattr(self.trainer, m_param)
                #     np.save(os.path.join(out_dir, f"{attr}.npy"), m_param_tensor.numpy())

                def run_vm():
                    # import numpy as np
                    loss = vm_exec()
                    self.tvm_device.sync()
                    # if self.overlap_comm_forward:
                    #     self.create_or_update_fw_outputs(loss[1:]) # skip loss
                    while isinstance(loss, (tuple, tvm.ir.container.Array, raf._core.value.TupleValue)):
                       loss = loss[0]
                    print("Loss: {}".format(loss.numpy()), flush=True)
                    # # DEBUG dump gradients
                    # m_model = self.model_w_loss

                    # assert m_model is not None
                    # # Get all parameters from RAF model.
                    # m_params = [p for p in dir(m_model) if p.startswith("model_")]
                    # out_dir = f"./grads/device_{device_id}/iter_{self.iteration_debug}"
                    # if not os.path.exists(out_dir):
                    #     os.makedirs(out_dir)
                    # if self.iteration_debug <= 5 or self.iteration_debug % 10 == 0:
                    #     for m_param in m_params:
                    #         # Get the corresponding parameter in PyTorch model.
                    #         attr = ""

                    #         for token in m_param[6:].split("_"):
                    #             if attr:
                    #                 attr += "_"
                    #             attr += token
                    #         m_param_tensor = getattr(m_model, m_param)
                    #         np.save(os.path.join(out_dir, f"{attr}.npy"), m_param_tensor.numpy())
                    #     m_params = [p for p in dir(self.trainer) if p.startswith("model_")]
                    #     for m_param in m_params:
                    #         # Get the corresponding parameter in PyTorch model.
                    #         attr = ""

                    #         for token in m_param[6:].split("_"):
                    #             if attr:
                    #                 attr += "_"
                    #             attr += token
                    #         m_param_tensor = getattr(self.trainer, m_param)
                    #         np.save(os.path.join(out_dir, f"{attr}.npy"), m_param_tensor.numpy())
                    # self.iteration_debug += 1
                    return loss

                self.executor = run_vm

        profile = kwargs.get("profile", 0)
        warmup = kwargs.get("warmup", 10)
        enable_lancet = kwargs.get("enable_lancet", False)
        if profile:
            if "LOAD_OPTIMIZED_MODULE_FROM" not in os.environ:
                logger.info("Running profile...")
                dctx = dist.get_context()
                run_setup(use_profile=False, enable_lancet=enable_lancet)
                # warmup
                for _ in range(warmup):
                    self.executor()
                raf.utils.profiler.get() # Clear existing results if any
                raf.utils.profiler.start()
                for _ in range(profile):
                    self.executor()
                raf.utils.profiler.stop()
                logger.info("Profile complete. Running schedule using profile...")
            else:
                logger.info(f"Load optimized module from {os.environ['LOAD_OPTIMIZED_MODULE_FROM']}, Skipping profile.")
            run_setup(use_profile=True, enable_lancet=enable_lancet)
        else:
            run_setup(use_profile=False, enable_lancet=False)
        super().bench_train_setup(device, **kwargs)

    def post_bench(self):
        """Reset distributed context settings after benchmark."""
        dctx = dist.get_context()
        dctx.enable_data_parallel = False
        dctx.zero_opt_level = 0
        super().post_bench()

    @staticmethod
    def get_tensor_size(t_list):
        """Calculate the total tensor size in MBs in the given list.

        Parameters
        ----------
        t_list: List[ndarray]
            A list of tensors.

        Returns
        -------
        float:
            The total tensor size in MBs.
        """
        tensors_size = 0.0
        for tensor in t_list:
            if tensor.dtype in ["float32", "int32"]:
                nbytes = 4
            elif tensor.dtype in ["float64", "int64"]:
                nbytes = 8
            elif tensor.dtype in ["float16", "int16"]:
                nbytes = 2
            elif tensor.dtype in ["int8", "uint8"]:
                nbytes = 1
            else:
                raise RuntimeError("Not support date type %s" % tensor.dtype)

            nsize = 1
            for shape in tensor.shape:
                nsize *= shape
            tensors_size += nsize * nbytes / 1048576.0

        return tensors_size

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
        data_parallel = kwargs.get("data_parallel", False)
        overlap_comm_forward = kwargs.get("overlap_comm_forward", False)
        if data_parallel:
            dctx = dist.get_context()
            dctx.enable_data_parallel = True
            dctx.overlap_comm_forward = overlap_comm_forward
            assert raf.build.with_distributed(), " RAF is not build with distributed support"
            assert device == "cuda", "data parallel should run with device of cuda"
            device = "cuda({})".format(dctx.local_rank)
        f_setup = self.bench_train_setup if train else self.bench_infer_setup(self)
        f_setup(device=device, **kwargs)

        # Warmup.
        for _ in range(10):
            self.executor()

        raf.utils.profiler.get()  # Clear existing results if any.
        raf.utils.profiler.start()
        self.executor()
        raf.utils.profiler.stop()
        return raf.utils.profiler.get()

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

        kwargs:
            See kwargs in bench_train_setup/bench_infer_setup for details.

        Returns
        -------
        float:
            The memory footprint in MBs.
        """
        f_setup = self.bench_train_setup if train else self.bench_infer_setup
        kwargs["dryrun"] = True
        f_setup(device=device, **kwargs)

        param_mem = self.get_tensor_size(self.vm_inputs)

        raf.utils.memory_profiler.reset()
        raf.utils.memory_profiler.start()
        timeit.repeat(self.executor, repeat=2, number=10)
        raf.utils.memory_profiler.stop()

        ret_map = raf.utils.memory_profiler.get_max_memory_info(raf.Device(self.device))
        max_mem_info = {k: v.value for k, v in ret_map.items()}

        print("Memory Profiling Summary")
        print("Parameter: %.2f MBs" % param_mem)
        print("Memory Pool (peak used): %.2f MBs" % max_mem_info["max_used"])
        print("Memory Pool (peak allocated): %.2f MBs" % max_mem_info["max_allocated"])
        print("#GarbageCollection: %.0f" % max_mem_info["num_gc"])

        return param_mem + (
            max_mem_info["max_used"] if show_used else max_mem_info["max_allocated"]
        )

    def get_memory_trace(self):
        """Get the memory trace. Note that trace is available after memory profiling,
        so make sure running profile_memory before calling this function.

        Returns
        -------
        str:
            The memory trace.
        """
        return raf.utils.memory_profiler.get_memory_trace(raf.Device(self.device))

    def check_gradient(self):
        """Check each gradient between RAF and PyTorch model."""
        import torch  # pylint: disable=import-outside-toplevel

        m_model = self.model_w_loss
        t_model = self.ref_bencher.model

        assert m_model is not None
        assert isinstance(t_model, torch.nn.Module), "Only support PyTorch models"

        # Get all parameters from RAF model.
        m_params = [p for p in dir(m_model) if p.startswith("model_")]

        stats = []
        for m_param in m_params:
            # Get the corresponding parameter in PyTorch model.
            grad = t_model
            if hasattr(grad, "module"):
                # handle the DDP case
                grad = grad.module

            attr = ""

            for token in m_param[6:].split("_"):
                if attr:
                    attr += "_"
                attr += token

                if hasattr(grad, attr):
                    grad = getattr(grad, attr)
                    attr = ""

            assert isinstance(
                grad, torch.Tensor
            ), "Expected torch.Tensor but got %s. Unmatched attr token: %s" % (
                type(grad),
                attr,
            )

            try:
                check(getattr(m_model, m_param), grad, atol=1e-4, rtol=1e-4)
                stats.append("%s ... passed" % m_param)
            except Exception as err:  # pylint: disable=broad-except
                atol = re.search(r"Max absolute difference: (.+)\n", str(err)).group(1)
                stats.append("%s ... failed (atol %s)" % (m_param, atol))

        for stat in sorted(stats):
            print(stat)

    def check_correctness(self, device="cuda", train=True, num_train_iter=1, check_gradient=False, **kwargs):
        """Check the correctness of the RAF model against to the reference model.

        Notes
        -----
        Correctness checking requires to run both RAF and reference models and both
        models will reserve device memory, so you might encounter out of memory error
        with large batch size. It is recommended to use batch size 1 in correctness checking.

        Parameters
        ----------
        device: str
            The target device. Default "cuda".

        train: bool
            When set to True, check the correctness of SGD training. Default True.

        check_gradient: bool
            Whether to check every gradient value.

        kwargs:
            sch_file: str
                The tuned log file path.

        Returns
        -------
        loss_pair: Tuple[float, float]
            A pair of loss from both RAF and original model.
        """
        assert self.ref_bencher is not None, "Reference bencher is required to check correctness"

        if train:
            overlap_comm_forward = kwargs.get("overlap_comm_forward", False)
            self.bench_train_setup(device, check_correctness=True, **kwargs)
            for i in range(num_train_iter):
                if not overlap_comm_forward or i == num_train_iter - 2:
                    out = self.bench_stmt()
                    print("RAF iter {} loss: {}".format(i, out.numpy()))
                else:
                    self.bench_stmt()
            self.post_bench()

            if overlap_comm_forward and num_train_iter == 1:
                out = self.first_iter_loss

            while isinstance(out, (tuple, tvm.ir.container.Array, raf._core.value.TupleValue)):
                out = out[0]

            self.ref_bencher.bench_train_setup(device, check_correctness=True, **kwargs)
            for i in range(num_train_iter):
                ref_out = self.ref_bencher.bench_stmt()
                print("Ref iter {} loss: {}".format(i, ref_out))
            self.ref_bencher.post_bench()
        else:
            self.bench_infer_setup(device, **kwargs)
            out = self.bench_stmt()
            self.post_bench()

            self.ref_bencher.bench_infer_setup(device, **kwargs)
            ref_out = self.ref_bencher.bench_stmt()
            self.ref_bencher.post_bench()
        try:
            check(out, ref_out, rtol=1e-4, atol=1e-4)
        except Exception as err:  # pylint: disable=broad-except
            print(err)
            # atol = re.search(r"Max absolute difference: (.+)\n", str(err))
            # shape = re.search(r"\(shapes (.+) mismatch\)\n", str(err))
            # if atol:
            #     atol = atol.group(1)
            #     logger.info("Correctness checking failure, atol %s", str(atol))
            # elif shape:
            #     shape = shape.group(1)
            #     logger.info("Correctness checking failure, shape mismatch: %s", str(shape))
            # else:
            #     logger.info("Correctness checking failure, unknown error:")
            #     logger.info(err)

        if check_gradient:
            assert train, "Cannot check gradient for forward inference"
            self.check_gradient()

        return (numpy(out), numpy(ref_out))

    def tune(
        self,
        sch_file,
        device="cuda",
        train=True,
        n_trials=lambda l: 300 * min(l, 100),
        only_tune_tasks_with_name=None,
        only_extract_tasks=False,
        **kwargs,
    ):
        """Use TVM auto-scheduler to tune the model.

        Parameters
        ----------
        sch_file: str
            The log file to dump the tuning records. If the file already contains tuning records,
            we use them to initialize the task scheduler and new records will be appended.

        device: str
            The target device. Default "cuda".

        train: bool
            When set to True, tuning the AutoDiff model. Default True.

        n_trials: Callable[[int], int] or int
            An integer of total number of measurement trials, or a function that determines
            the total number of measurement trials by taking the task number.
            Default is at maximum 30k = 300 * min(100, task_num).

        only_tune_tasks_with_name: Optional[List[str]]
            When specify with a list of name tokens, only the tasks with the tokens in their names
            will be tuned.

        only_extract_tasks: bool
            Whether to extract and print tasks only without actual tuning them.

        kwargs:
            Used to setup the model to be tuned. See kwargs in `bench_train_setup` for details.
        """
        if "use_interpreter" in kwargs:
            assert not kwargs["use_interpreter"], "Only support tuning with VM but not interpreter"
        data_parallel = kwargs.get("data_parallel", False)
        overlap_comm_forward = kwargs.get("overlap_comm_forward", False)
        if data_parallel:
            dctx = dist.get_context()
            dctx.enable_data_parallel = True
            dctx.overlap_comm_forward = overlap_comm_forward
            assert raf.build.with_distributed(), " RAF is not build with distributed support"
            # use the user's specified device for tuning

        # Use memory profiler to extract tuning tasks because memory profiler
        # will ignore the op compilation failture due to no valid schedule and
        # #skip the op execution.
        f_setup = self.bench_train_setup if train else self.bench_infer_setup
        f_setup(device=device, dryrun=True, is_tuning=True, **kwargs)

        run_tuning(
            self.vm,
            device,
            self.vm_inputs,
            sch_file,
            n_trials=n_trials,
            only_tune_tasks_with_name=only_tune_tasks_with_name,
            only_extract_tasks=only_extract_tasks,
        )
