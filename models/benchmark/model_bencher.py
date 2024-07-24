"""
The module of wrapper classes to benchmark models.
"""
# pylint: disable=unused-argument, too-many-arguments
import timeit
import os

import raf
from raf import distributed as dist
import numpy as np

from .utils import func_timer


class ModelBencherBase:
    """The class to benchmark a model.

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
        self.model = model
        self.input_shape = input_shape
        self.dataloader = dataloader
        # self.args = args
        self.dy = dy
        # self.y_true = y_true
        self.kwargs = kwargs

        self.executor = None

    def bench(self, device="cuda", warmup=10, number=100, train=True, **kwargs):
        """Benchmark the model performance.

        Parameters
        ----------
        device: str
            The target device. Default "cuda".

        warmup: int
            The repeated times to warm-up the model. Default 10.

        number: int
            The repated times to evaluate the performance. The final latency
            is the average of all repeated runs. Default 100.

        train: bool
            When set to True, evaluating the latency of SGD training. Default True.

        Returns
        -------
        ret: float
            The average latency.
        """
        if train:
            t_setup = self.bench_train_setup
        else:
            t_setup = self.bench_infer_setup
        t_setup(device, warmup=warmup, number=number, **kwargs)

        t_stmt = self.bench_stmt

        # Warm-up.
        func_timer("Warmup")(timeit.repeat)(t_stmt, repeat=1, number=warmup)
        self.post_warmup()

        # Benchmark. The given number will be adjusted to 5*n if it is not dividable.
        unit_number = 1
        repeat = number // unit_number
        if repeat == 0:
            # If the number is too small, then change unit number to 1.
            repeat = unit_number
            unit_number = 1

        def run_stmt():
            for _ in range(unit_number):
                t_stmt()
            self.post_bench_timed()

        profile_bench = kwargs.get("profile_bench", False)
        if profile_bench:
            raf.utils.profiler.get() # Clear existing results if any
            raf.utils.profiler.start()

        raf.utils.profiler.cuda_profiler_start()
        t_time = timeit.repeat(run_stmt, repeat=repeat, number=1)
        raf.utils.profiler.cuda_profiler_stop()

        if profile_bench:
            raf.utils.profiler.stop()
            raf.utils.profiler.dump(filename=profile_bench.replace("%p", str(os.getpid())))

        # Based on https://docs.python.org/2/library/timeit.html#timeit.Timer.repeat,
        # min should be taken rather than the average.
        # The benchmark in HuggingFace transformers also adopts this method:
        # https://github.com/huggingface/transformers/blob/3c0c699ffd7f434451b1e2a483aa7458633b35cc/src/transformers/benchmark/benchmark.py#L209
        # However, we still feel median is more realistic for our benchmark.
        self.post_bench()
        return 1e3 * np.median(t_time) / unit_number

    def visualize_dag(self, output_file_name, device="cuda", train=True, dump_as_text=False, **kwargs):
        raise NotImplementedError

    def bench_infer_setup(self, device="cuda", **kwargs):
        """Setup the model to benchmark forward inference.

        Parameters
        ----------
        device: str
            The target device. Default "cuda".
        """

    def bench_train_setup(self, device="cuda", **kwargs):
        """Setup the model to benchmark backward training.

        Parameters
        ----------
        device: str
            The target device. Default "cuda".
        """

    def post_warmup(self):
        """Post-processing after warmup."""

    def post_bench_timed(self):
        """Post-processing after running benchmark statement (time included in benchmark)."""
        pass

    def post_bench(self):
        """Post-processing after benchmark."""
        raf.distributed.RemoveCommunicator()

    def bench_stmt(self):
        """Benchmark the executor. Note that we should make sure self.executor
        is not None, but since we benchmark the runtime of this function, we want
        to avoid any statements in addition to running the model.

        Returns
        -------
        out: Any
            The output.
        """
        return self.executor()  # pylint: disable=not-callable

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
        raise NotImplementedError
