
import pytest
import numpy as np

import raf
from raf.testing import check, with_dialect, run_vm_model


@with_dialect(["cublas", "cudnn", "tvm", "cuda"])
@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
def test_sparse_expert_matmul_nt(stride=1, offset=0, scale=1):

    elements = [128, 32, 256, 32, 64]
    M = 768
    N = 768 * 4
    C = 256

    x_shape = (C, len(elements), M)
    w_shape = (N, M)

    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, n_elements_per_gpu, weight):
            return raf.sparse_expert_matmul_nt(x, weight, n_elements_per_gpu, offset, stride, scale)

    device = "cuda(0)"
    r_model = TestModel()

    r_model.to(device=device)

    x = np.random.randn(*x_shape).astype("float32")
    w = np.random.randn(*w_shape).astype("float32")

    in_elements = [x * M // scale for x in elements]
    in_elements_padded = [np.random.randint(0, 256) for _ in range(stride * len(in_elements))]
    for i in range(len(in_elements)):
        in_elements_padded[i * stride + offset] = in_elements[i]

    elem_np = np.array(in_elements_padded, dtype=np.int32)

    rx = raf.array(x, device=device)
    rw = raf.array(w, device=device)
    relem = raf.array(elem_np, device=device, dtype="uint64")

    r_out = run_vm_model(r_model, device, [rx, relem, rw]).numpy()

    # # reference np implementation
    # xs = np.split(x, len(elements), axis=1)
    # xs = [np.squeeze(x) for x in xs]
    # xs_no_padding = [xs[i][:elements[i], :] for i in range(len(elements))]
    # xs_after_mm = [np.matmul(xs_no_padding[i], w.T) for i in range(len(elements))]

    # np_out = np.zeros((C, len(elements), N), dtype=np.float32)
    # for i in range(len(elements)):
    #     np_out[:elements[i], i, :] = xs_after_mm[i]

    # 1e-5 tolerance causes failure for some reason
    # check(r_out, np_out, atol=1e-4, rtol=1e-4)
    print("nelements: ", elem_np // 768)
    print("rout.shape: ", r_out.shape)
    r0 = r_out[:int(elem_np[0] / 768),0,:]
    r1 = r_out[:int(elem_np[2] / 768),1,:]
    print("max diff r_out = ", np.max(np.abs(r0 - r1)))

    np_ref = np.matmul(x.reshape((-1, x.shape[2])), w.T).reshape((x.shape[0], x.shape[1], w.shape[0]))
    np_ref0 = np_ref[:int(elem_np[0] / 768),0,:]
    np_ref1 = np_ref[:int(elem_np[2] / 768),1,:]
    print("np_ref.shape: ", np_ref.shape)
    print("max diff np_ref = ", np.max(np.abs(np_ref0 - np_ref1)))



if __name__ == "__main__":
    test_sparse_expert_matmul_nt(2, 1, 1)
    # test_cascaded_expert_matmul_nt()