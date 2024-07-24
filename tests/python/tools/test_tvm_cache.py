# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import os
# os.environ["RAF_PERSIST_CACHE"] = "1"
# os.environ["RAF_CACHE_DIR"] = "./test_caching/raf_cache"

import raf
from raf.testing import mlp, check, randn_torch, run_vm_model



def test_dump_cache(device, dump_cache=False, load_cache=False):
    if load_cache:
        raf.cache.LoadAllTVMCaches("./test_caching", "rank0")
    config = (784, 10, 256, 256)
    m_model, t_model = mlp.get_model(config)
    m_model.to(device=device)
    t_model.to(device=device)
    m_optimizer = raf.optim.sgd.with_sgd(learning_rate=0.1, momentum=0.01)(m_model)[0]
    t_optimizer = torch.optim.SGD(t_model.parameters(), lr=0.1, momentum=0.01)
    m_dy, t_dy = randn_torch((), device=device, requires_grad=False)
    m_in, t_in = mlp.get_input(config, batch_size=1, device=device)
    m_loss = run_vm_model(m_optimizer, device, [m_dy, *m_in])[0]
    t_optimizer.zero_grad()
    t_loss = t_model(*t_in)
    t_loss.backward(t_dy)
    t_optimizer.step()
    check(m_loss, t_loss, atol=1e-4, rtol=1e-4)
    mlp.check_params(m_model, t_model, atol=1e-4, rtol=1e-4)
    if dump_cache:
        raf.cache.DumpAllTVMCaches("./test_caching")



if __name__ == "__main__":
    test_dump_cache(device="cuda", load_cache=True)

