# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=too-many-locals,too-many-arguments,protected-access,no-self-use, line-too-long, superfluous-parens
from typing import Dict, List
import pytest
import numpy as np
import tvm

from raf._core.device import Device
import numpy as np
import pytest
import raf
from raf.testing import check, run_vm_model, with_dialect

@with_dialect(["cuda"])
@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("shape", [(), (1,), (64, 128), (128, 256)])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("device",["cuda(0)", "cuda(1)"])
def test_cuda_zeros(shape, dtype, device):
    class InitOpModel(raf.Model):
        def build(self, shape, dtype, device):
            self._shape = shape
            self._dtype = dtype
            self._device = device

        @raf.model.trace
        def forward(self):
            return raf.zeros(shape=self._shape, dtype=self._dtype, device=self._device)

    with Device(device):
        model = InitOpModel(shape, dtype, device)
        model.to(device=device)
        v_y = run_vm_model(model, device, [])
        n_y = np.zeros(shape=shape, dtype=dtype)
        check(v_y, n_y)

if __name__ == "__main__":
    pytest.main([__file__])
