# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=too-many-locals,too-many-arguments,protected-access,no-self-use, line-too-long, superfluous-parens
from typing import Dict, List
import pytest
import numpy as np
import tvm

import raf
from raf._core.device import Device
from raf.testing import check, with_dialect
from raf._core.ir_ext import extended_var
from raf.ir import ScopeBuilder
from raf._core.ndarray import get_ndarray_handle, ndarray
from raf.testing.utils import get_vm_executor
from raf.utils.tuner import run_tuning
from raf._ffi.model import RunModel
from raf._ffi.pass_ import InferType


import argparse

parser = argparse.ArgumentParser(description='Tune tvm layernorm dx op.')
parser.add_argument('--shape', required=True, type=int, nargs="+", help='input shape')
parser.add_argument('--eps', required=True, type=float, help='eps')
parser.add_argument('--output', '-o', type=str, default="./tvm_layernorm_dx_schedule.json",
                    help='output schedule path.')

from raf.utils.tuner import tune_layer_norm_dx

if __name__ == "__main__":
    args = parser.parse_args()
    input_shape = tuple(args.shape)
    print("Tuning with input shape: ", input_shape, ", eps: ", args.eps)
    space_dict = {
        "batch_size": [input_shape[0]],
        "seq_length": [input_shape[1]],
        "hidden_size": [input_shape[2]],
        "eps": [args.eps],
    }
    tune_layer_norm_dx(args.output, space_dict=space_dict)
