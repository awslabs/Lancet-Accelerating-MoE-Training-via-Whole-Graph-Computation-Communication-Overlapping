"""
The CLI entry.
"""
from pathlib import Path
import argparse
import json
import numpy as np

from . import get_model_bencher


def create_config():
    """Create the CLI configuration.

    Returns
    -------
    ArgumentParser:
        The parsed commandline arguments.
    """
    # Common options.
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--framework", type=str, help="Framework name")
    common_parser.add_argument("--model", type=str, help="Model name")
    common_parser.add_argument("--batch", type=int, default=32, help="Batch size")
    common_parser.add_argument("--dtype", type=str, default="float32", help="Model dtype")
    common_parser.add_argument(
        "--workload", type=str, help="Workload/image size/sqeuence length in JSON format"
    )
    common_parser.add_argument("--device", type=str, default="cuda", help="The target device")
    common_parser.add_argument(
        "--infer", default=False, action="store_true", help="Benchmark inference mode"
    )
    common_parser.add_argument(
        "--log", default=None, help="If present, the result will be appended to the log file"
    )
    common_parser.add_argument(
        "--amp", action="store_true", default=False, help="Enable automatic mixed precision"
    )
    common_parser.add_argument(
        "--optimizer", type=str, default="SGD", help="Optimizer, only surport SGD and LANS"
    )

    # RAF sepcific options (no effect in other frameworks).
    common_parser.add_argument(
        "--data-parallel", action="store_true", help="Enable data parallel (RAF Only)"
    )
    common_parser.add_argument(
        "--zero-opt", type=int, default=0, help="ZeRO optimization level (RAF Only)"
    )
    common_parser.add_argument(
        "--disable-fuse", action="store_true", help="Disable fusion (RAF Only)"
    )
    common_parser.add_argument(
        "--sch-file",
        default=None,
        help="The schedule file for benchmarking (read) or tuning (write) (RAF Only)",
    )

    # PyTorch specific options (no effect in other frameworks).
    common_parser.add_argument(
        "--ltc",
        default=None,
        help="Benchmark using LazyTensorCore (xla or razor) (PyTorch Only)",
    )

    # Benchmark modes.
    mode_parser = argparse.ArgumentParser()
    subprasers = mode_parser.add_subparsers(dest="mode", help="Execution modes")
    subprasers.add_parser("bench", parents=[common_parser], help="Benchmark latency")
    pm_parser = subprasers.add_parser(
        "profile_memory",
        parents=[common_parser],
        help="Profile the peak memory",
    )
    pm_parser.add_argument(
        "--allocated",
        action="store_true",
        help="If present, trace the total allocated memory instead of used memory "
        "(only applicable to RAF models)",
    )

    # RAF specific execution modes.
    subprasers.add_parser(
        "check_correctness",
        parents=[common_parser],
        help="Check the loss correctness (RAF Only)",
    )
    subprasers.add_parser(
        "profile_latency",
        parents=[common_parser],
        help="Profile the latency of each op (RAF Only)",
    )
    subprasers.add_parser(
        "trace_memory",
        parents=[common_parser],
        help="Trace memory of a RAF model (RAF Only)",
    )
    tune_parser = subprasers.add_parser(
        "tune", parents=[common_parser], help="Tune Model with Ansor (RAF Only)"
    )
    tune_parser.add_argument("--ntrials", type=int, help="Trial number for tuning")
    tune_parser.add_argument(
        "--only-tune-tasks",
        type=str,
        help="Only tune the tasks which specified names. "
        "Use comma to specify multiple names if needed",
    )
    tune_parser.add_argument(
        "--print-task-only", action="store_true", help="Only extract and print tasks without tuning"
    )

    return mode_parser.parse_args()


def main():
    """The main entry."""
    args = create_config()

    # Parse CLI configurations
    try:
        shape = json.loads(args.workload) if args.workload is not None else None
    except json.decoder.JSONDecodeError:
        raise RuntimeError("Invalid input workload: %s" % args.workload)
    train = not args.infer

    if args.sch_file is not None:
        print("Schedule file is pointed to %s/%s" % (Path.cwd(), args.sch_file))

    bencher = get_model_bencher(
        args.framework,
        args.model,
        batch_size=args.batch,
        dtype=args.dtype,
        shape=shape,
        include_orig_model=args.mode == "check_correctness",
    )
    assert bencher is not None, "Failed to load model for %s/%s" % (args.framework, args.model)

    ret = None
    if args.mode == "bench":
        ret = "%.2f" % bencher.bench(
            args.device,
            train=train,
            disable_fuse=args.disable_fuse,
            amp=args.amp,
            optimizer=args.optimizer,
            sch_file=args.sch_file,
            data_parallel=args.data_parallel,
            zero_opt=args.zero_opt,
            ltc=args.ltc,
        )
    elif args.mode == "check_correctness":
        assert args.framework == "", "Expected , but got " % args.framework
        out, ref_out = bencher.check_correctness(
            args.device,
            disable_fuse=args.disable_fuse,
            amp=args.amp,
            optimizer=args.optimizer,
        )
        ret = str(np.max(np.abs(out - ref_out)))
    elif args.mode == "profile_latency":
        profile_log = "profile_%s_bs_%d_fuse_%d%s_%s.log" % (
            args.model,
            args.batch,
            args.disable_fuse,
            "_amp" if args.amp else "",
            args.optimizer,
        )
        result = bencher.profile_latency(
            args.device,
            train=train,
            disable_fuse=args.disable_fuse,
            amp=args.amp,
            optimizer=args.optimizer,
            sch_file=args.sch_file,
        )
        with open(profile_log, "w") as filep:
            json.dump(result, filep, indent=4)
    elif args.mode == "trace_memory":
        assert args.framework == "", "Expected , but got " % args.framework
        bencher.profile_memory(
            args.device,
            train=train,
            disable_fuse=args.disable_fuse,
            amp=args.amp,
            optimizer=args.optimizer,
        )
        trace_log = "trace_%s_bs_%d_fuse_%d_amp_%s_%s.log" % (
            args.model,
            args.batch,
            args.disable_fuse,
            args.amp,
            args.optimizer,
        )
        with open(trace_log, "w") as filep:
            filep.write(bencher.get_memory_trace())
    elif args.mode == "profile_memory":
        ret = "%.2f" % bencher.profile_memory(
            not args.allocated,
            args.device,
            train=train,
            disable_fuse=args.disable_fuse,
            amp=args.amp,
            ltc=args.ltc,
            optimizer=args.optimizer,
        )
    elif args.mode == "tune":
        assert args.framework == "", "Expected , but got " % args.framework
        only_tune_tasks = None if not args.only_tune_tasks else args.only_tune_tasks.split(",")
        n_trials = args.ntrials if args.ntrials is not None else lambda l: 128 * min(l, 100)
        bencher.tune(
            sch_file=args.sch_file if args.sch_file is not None else "tuning.json",
            device=args.device,
            train=train,
            n_trials=n_trials,
            only_tune_tasks_with_name=only_tune_tasks,
            only_extract_tasks=args.print_task_only,
            disable_fuse=args.disable_fuse,
            amp=args.amp,
            optimizer=args.optimizer,
        )
    else:
        raise RuntimeError("Unrecognized mode: %s" % args.mode)

    if ret is not None:
        if args.log is not None:
            with open(args.log, "a") as filep:
                filep.write(ret)
        else:
            print(ret)


if __name__ == "__main__":
    main()
