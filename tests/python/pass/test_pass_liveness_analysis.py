# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name, no-self-use, too-many-locals, unused-variable, protected-access
# pylint: disable=too-many-arguments
import pytest
import raf
from raf._lib import tvm, relay
from raf.ir import ScopeBuilder
from raf._ffi.pass_ import InferType, LivenessAnalysis, ManifestAlloc
from raf.testing import randn
from raf._core.ir_ext import extended_var


def verify_live_in_set(mod, expected):
    mod = InferType()(mod)

    # Check liveness analysis result.
    ret = LivenessAnalysis(mod)

    ret = {key.name_hint: {vkey: {v.name_hint for v in var_list} for vkey, var_list in var_dict.items()} for key, var_dict in ret.items()}

    def intersect_streamvset(svset1, svset2):
        # svset1 - svset2
        difference = {}
        for key, stream_dict in svset1.items():
            if key not in svset2:
                difference[key] = {}
            else:
                for exp_stream, exp_vset in stream_dict.items():
                    if exp_stream not in svset2[key]:
                        if key not in difference:
                            difference[key] = {}
                        difference[key][exp_stream] = []
                    else:
                        for var in exp_vset:
                            if var not in svset2[key][exp_stream]:
                                if key not in difference:
                                    difference[key] = {}
                                if exp_stream not in difference[key]:
                                    difference[key][exp_stream] = []
                                difference[key][exp_stream].append(var)
        return difference

    missed = intersect_streamvset(expected, ret)
    extra = intersect_streamvset(ret, expected)

    if missed or extra:
        print("IR:\n%s" % raf.ir.AsText(mod))
        print("Live in sets:")
        for key, stream_dict in ret.items():
            for stream, var_list in stream_dict.items():
                print("key %s, stream %s: %s" % (key, stream, ",".join(var_list)))

        print("\nMissed items")
        for key, stream_dict in missed.items():
            if not stream_dict:
                print("Missed key %s" % key)
            for stream, var_list in stream_dict.items():
                if not var_list:
                    print("Missed key %s" % key)
                else:
                    print("Missed live in of %s: %s" % (key, ",".join(var_list)))

        print("\nExtra items")
        for key, stream_dict in extra.items():
            if not stream_dict:
                print("Extra key %s" % key)
            for stream, var_list in stream_dict.items():
                if not var_list:
                    print("Extra key %s" % key)
                else:
                    print("Extra live in of %s: %s" % (key, ",".join(var_list)))
        assert False, "Live in set mismatch"
    print("Live in set matches.")


def test_basic():
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, param_0, param_1, param_2, param_3):
            t_0 = raf.add(param_0, param_0)  # a1
            t_1 = raf.add(t_0, param_1)  # a2
            t_2 = raf.add(t_1, param_2)  # a3
            t_3 = raf.add(t_2, t_0)  # a4
            t_4 = raf.add(t_3, param_3)  # a5
            return t_4  # n_1

    device = "cpu"
    shape = (5, 5)
    model = Model()
    model.infer_mode()
    m_a, _ = randn(shape, device=device)
    m_b, _ = randn(shape, device=device)
    m_c, _ = randn(shape, device=device)
    m_d, _ = randn(shape, device=device)
    args = [m_a, m_b, m_c, m_d]

    expected = {
        "n_0": {},
        "a1": {-1: ["param_0", "param_1", "param_2", "param_3"]},
        "a2": {-1: ["t_0", "param_1", "param_2", "param_3"]},
        "a3": {-1: ["t_0", "t_1", "param_2", "param_3"]},
        "a4": {-1: ["t_0", "t_2", "param_3"]},
        "a5": {-1: ["t_3", "param_3"]},
        "n_1": {-1: ["t_4"]},
    }

    mod = model._internal(*args).mod
    verify_live_in_set(mod, expected)


def test_multi_outs():
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, param_0, param_1, param_2, param_3, param_4):
            t_0 = raf.relu(param_0)  # a1
            res = raf.batch_norm_train(t_0, param_3, param_4, param_1, param_2, 0.1, 1e-5)  # a2
            t_1 = res[0]  # a3
            t_2 = res[1]
            t_3 = res[2]
            t_4 = raf.relu(t_1)  # a4
            t_5 = raf.relu(t_4)  # a5
            return t_5  # n_1

    model = Model()
    model.infer_mode()

    device = "cpu"
    shape = (16, 3, 224, 224)
    stats_shape = [shape[1]]
    m_x, _ = randn(shape, device=device)
    m_m, _ = randn(stats_shape, device=device)
    m_v, _ = randn(stats_shape, positive=True, device=device)
    m_w, _ = randn(stats_shape, device=device)
    m_b, _ = randn(stats_shape, device=device)
    args = [m_x, m_m, m_v, m_w, m_b]

    expected = {
        "n_0": {},
        "a1": {-1: ["param_0", "param_1", "param_2", "param_3", "param_4"]},
        "a2": {-1: ["t_0", "param_1", "param_2", "param_3", "param_4"]},
        "a3": {-1: ["t_1"]},
        "a4": {-1: ["t_1"]},
        "a5": {-1: ["t_4"]},
        "n_1": {-1: ["t_5"]},
    }

    mod = model._internal(*args).mod
    verify_live_in_set(mod, expected)


def test_tuple_input():
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, tup):
            x = tup[0]  # a1
            y = tup[1]  # a2
            t_0 = raf.add(x, y)  # a3
            return t_0  # n_1

    model = Model()
    model.infer_mode()

    device = "cpu"
    shape = (5, 5)
    m_a, _ = randn(shape, device=device)
    m_b, _ = randn(shape, device=device)
    args = [(m_a, m_b)]

    expected = {
        "n_0": {},
        "a1": {-1: ["param_0", "param_1"]},
        "a2": {-1: ["param_0", "param_1"]},
        "a3": {-1: ["param_0", "param_1"]},
        "n_1": {-1: ["t_0"]},
    }

    mod = model._internal(*args).mod
    verify_live_in_set(mod, expected)


def test_unused_tuple():
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, tup):
            x = tup[0]  # a1
            t_0 = raf.add(x, x)  # a2
            t_1 = raf.concatenate(tup)  # a3
            ret = (t_0, t_1)  # a4
            return ret  # n_1

    model = Model()
    model.infer_mode()

    device = "cpu"
    shape = (5, 5)
    m_a, _ = randn(shape, device=device)
    m_b, _ = randn(shape, device=device)
    args = [(m_a, m_b)]

    # There won't be a tgi_1 because it is never be used.
    expected = {
        "n_0": {},
        "a1": {-1: ["param_0", "param_1"]},
        "a2": {-1: ["param_0", "param_1"]},
        "a3": {-1: ["param_0", "param_1", "t_0"]},
        "a4": {-1: ["t_0", "t_1"]},
        "n_1": {-1: ["t_0", "t_1"]},
    }

    mod = model._internal(*args).mod
    verify_live_in_set(mod, expected)


def test_direct_assign():
    sb = ScopeBuilder()
    p0 = raf.ir.var("p0", shape=(10, 10))
    a_1 = sb.let("a1", raf.ir.op.relu(p0))
    a_2 = sb.let("a2", a_1)
    a_3 = sb.let("a3", raf.ir.op.relu(a_2))
    sb.ret(a_3)
    mod = tvm.IRModule.from_expr(relay.Function([p0], sb.get()))

    expected = {
        "n_0": {},
        "a1": {-1: ["param_0"]},
        "a2": {-1: ["t_0"]},
        "a3": {-1: ["t_0"]},
        "n_1": {-1: ["t_1"]},
    }
    verify_live_in_set(mod, expected)


def test_reshape():
    shape = (10, 10)

    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            t_0 = raf.relu(x)
            t_1 = raf.reshape(t_0, (shape[0] * shape[1],))
            t_2 = raf.relu(t_1)
            return t_2

    model = Model()
    model.infer_mode()

    device = "cpu"
    m_x, _ = randn(shape, device=device)
    args = [m_x]

    expected = {
        "n_0": {},
        "a1": {-1: ["param_0"]},
        "a2": {-1: ["t_0"]},
        "a3": {-1: ["t_0"]},
        "n_1": {-1: ["t_1"]},
    }

    mod = model._internal(*args).mod
    verify_live_in_set(mod, expected)

class ANFBuilder:
    def __init__(self):
        self.scope_builder = ScopeBuilder()
        self.operators = {}

    def get_operator(self, op_name: str) -> tvm.ir.Op:
        if op_name not in self.operators:
            self.operators[op_name] = raf._ffi.op.GetOp(f"raf.op.{op_name}")
        return self.operators[op_name]

    def const(self, value):
        return raf.ir.const(value)

    def make_tuple(self, fields, name=""):
        return self.scope_builder.let(name, tvm.relay.Tuple(fields))

    def get_tuple_item(self, tup, index, name=""):
        return self.scope_builder.let(name, tvm.relay.TupleGetItem(tup, index))

    def call(self, op_name: str, args, name="") -> tvm.relay.Var:
        return self.scope_builder.let(name, tvm.relay.Call(self.get_operator(op_name), args))

    def ret(self, body: tvm.relay.Expr) -> tvm.relay.Expr:
        self.scope_builder.ret(body)
        return self.scope_builder.get()

def test_basic_multistream():
    def test_func():
        builder = ANFBuilder()
        x = extended_var("x", shape=(5, 5))
        y = extended_var("y", shape=(5, 5))
        z = extended_var("z", shape=(5, 5))
        l = extended_var("l", shape=(5, 5))
        builder.call("set_stream", [builder.const(0), builder.const(1)], name="set_stream_0") # n0
        a0 = builder.call("add", [x, y], name="add_0") # t0
        a1 = builder.call("add", [a0, z], name="add_1") # t1
        builder.call("set_stream", [builder.const(0), builder.const(4)], name="set_stream_1") # n1
        a2 = builder.call("add", [a0, l], name="add_2") # t2
        ret = builder.make_tuple([a1, a2], name="ret") # n3
        return relay.Function([x, y, z, l], builder.ret(ret))

    func = test_func()
    mod = tvm.IRModule.from_expr(func)
    expected = {
        "set_stream_0": {1: ["param_0", "param_1", "param_2"], 4: ["param_3"]},
        "add_0": {1: ["param_0", "param_1", "param_2"], 4: ["param_3"]},
        "add_1": {1: ["t_0", "param_2"], 4: ["t_0", "param_3"]},
        "set_stream_1": {1: [], 4: ["t_0", "t_1", "param_3"]},
        "add_2": {1: [], 4: ["t_0", "t_1", "param_3"]},
        "ret": {1: [], 4: ["t_1", "t_2"]},
        "n_3": {1: [], 4: ["t_1", "t_2"]},
        "n_2": {}
    }
    verify_live_in_set(mod, expected)

def test_basic_multistream_1():
    def test_func():
        builder = ANFBuilder()
        x = extended_var("x", shape=(5, 5))
        builder.call("set_stream", [builder.const(0), builder.const(1)], name="set_stream_0") # n0
        a0 = builder.call("relu", [x], name="relu0") # t0
        a0t = builder.make_tuple([a0], name="relu0t")
        a1 = builder.call("relu", [a0], name="relu1") # t1
        builder.call("set_stream", [builder.const(0), builder.const(4)], name="set_stream_1") # n1
        a2 = builder.call("_allreduce", [a0t, builder.const("sum")], name="allreduce") # t2
        builder.call("set_stream", [builder.const(0), builder.const(1)], name="set_stream_2") # n2
        ret = builder.make_tuple([a1, a2], name="ret") # n4
        return relay.Function([x], builder.ret(ret))

    func = test_func()
    mod = tvm.IRModule.from_expr(func)
    expected = {
        "set_stream_0": {1: ["param_0"], 4: []},
        "relu0": {1: ["param_0"], 4: []},
        "relu0t": {1: ["t_0"], 4: ["t_0"]},
        "relu1": {1: ["t_0"], 4: ["t_0"]},
        "set_stream_1": {1: ["t_1"], 4: ["t_0"]},
        "allreduce": {1: ["t_1"], 4: ["t_0"]},
        "set_stream_2": {1: ["t_1", "t_2"], 4: []},
        "n_4": {1: ["t_1", "t_2"], 4: []},
        "ret": {1: ["t_1", "t_2"], 4: []},
        "n_3": {},
    }
    verify_live_in_set(mod, expected)

def test_multistream_with_tuple():
    def test_func():
        builder = ANFBuilder()
        x = extended_var("x", shape=(5, 5))
        y = extended_var("y", shape=(5, 5))
        z = extended_var("z", shape=(5, 5))
        sc = extended_var("send_counts", shape=(16,), dtype="uint64")
        builder.call("set_stream", [builder.const(0), builder.const(1)], name="set_stream_0") # n0
        a0 = builder.call("add", [x, y], name="add_0") # t0
        a1 = builder.call("add", [a0, z], name="add_1") # t1
        builder.call("set_stream", [builder.const(0), builder.const(4)], name="set_stream_1") # n1
        a2 = builder.make_tuple([a0], name="tup_0") # t0
        a3 = builder.make_tuple([sc], name="tup_1") # param_3
        a4 = builder.call("_all_to_allv", [a2, a3], name="all_to_allv") # ms_0
        a5 = builder.get_tuple_item(a4, 0, name="tgi_0") # t2
        ret = builder.make_tuple([a1, a5], name="ret") # ms_1
        return relay.Function([x, y, z, sc], builder.ret(ret))

    func = test_func()
    mod = tvm.IRModule.from_expr(func)
    expected = {
        "set_stream_0": {1: ["param_0", "param_1", "param_2"], 4: ["param_3"]},
        "add_0": {1: ["param_0", "param_1", "param_2"], 4: ["param_3"]},
        "add_1": {1: ["t_0", "param_2"], 4: ["t_0", "param_3"]},
        "set_stream_1": {1: [], 4: ["t_0", "t_1", "param_3"]},
        "tup_0": {1: [], 4: ["t_0", "t_1", "param_3"]},
        "tup_1": {1: [], 4: ["t_0", "t_1", "param_3"]},
        "all_to_allv": {1: [], 4: ["t_0", "t_1", "param_3"]},
        "tgi_0": {1: [], 4: ["t_1", "t_2"]},
        "ret": {1: [], 4: ["t_1", "t_2"]},
        "n_3": {1: [], 4: ["t_1", "t_2"]},
        "n_2": {}
    }
    verify_live_in_set(mod, expected)



def test_manifest_alloc_compatible():
    def test_func():
        add_op = raf._ffi.op.GetOp("raf.op.add")
        null = raf.ir.const(None)

        x = relay.var("x", shape=(5, 5))
        y = relay.var("y", shape=(5, 5))
        a0 = relay.var("a0")
        a1 = relay.var("a1")
        a2 = relay.var("a2")
        a3 = relay.var("a3")
        a4 = relay.var("a4")
        a5 = relay.var("a5")
        a6 = relay.var("a6")
        a7 = relay.var("a7")

        let7 = relay.Let(a7, a1, a7)
        let6 = relay.Let(a6, raf.ir.op.vm_invoke_op(a2, a4, a5), let7)
        let5 = relay.Let(a5, relay.Tuple((a1,)), let6)
        # Test both binded and non-binded constants
        let4 = relay.Let(a4, relay.Tuple((x, y, a3, null)), let5)
        let3 = relay.Let(a3, null, let4)
        let2 = relay.Let(a2, add_op, let3)
        let1 = relay.Let(a1, raf.ir.op.vm_alloc_tensor(a0, [5, 5], "float32", [5, 5]), let2)
        let0 = relay.Let(a0, raf.ir.op.vm_alloc_storage(100, 64, 1, 0), let1)
        # pylint: disable=line-too-long
        # fn (%x: Tensor[(5, 5), float32], %y: Tensor[(5, 5), float32]) {
        #   let %a0 = raf.op.vm.alloc_storage(int64(100), int64(64), int64(1), int64(0), str"float32");
        #   let %a1 = raf.op.vm.alloc_tensor(%a0, TupleValue([int64(5), int64(5)]), str"float32", TupleValue([int64(5), int64(5)]));
        #   let %a2 = raf.op.add;
        #   let %a3 = nullptr;
        #   let %a4 = (%x, %y, %a3, nullptr);
        #   let %a5 = (%a1,);
        #   let %a6 = raf.op.vm.invoke_op(%a2, %a4, %a5);
        #   let %a7 = %a1;
        #   %a7
        # }
        # pylint: enable=line-too-long

        return relay.Function([x, y], let0)

    # Note that a3 will be inlined after InferType.
    expected = {
        "n_1": {},
        "a0": {-1: ["param_0", "param_1"]},
        "a1": {-1: ["param_0", "param_1", "t_0"]},
        "a2": {-1: ["param_0", "param_1", "t_1"]},
        "a4": {-1: ["param_0", "param_1", "t_1"]},
        "a5": {-1: ["param_0", "param_1", "t_1"]},
        "a6": {-1: ["param_0", "param_1", "t_1"]},
        "a7": {-1: ["t_1"]},
        "n_2": {-1: ["t_1"]},
    }

    func = test_func()
    mod = tvm.IRModule.from_expr(func)
    verify_live_in_set(mod, expected)


def test_after_manifest_alloc():
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, param_0, param_1, param_2):
            t_0 = raf.add(param_0, param_0)  # a1
            t_1 = raf.add(t_0, param_1)  # a2
            t_2 = raf.add(t_1, param_2)  # a3
            return t_2  # n_1

    device = "cpu"
    shape = (5, 5)
    model = Model()
    model.infer_mode()
    m_a, _ = randn(shape, device=device)
    m_b, _ = randn(shape, device=device)
    m_c, _ = randn(shape, device=device)
    args = [m_a, m_b, m_c]

    mod = model._internal(*args).mod
    mod = InferType()(mod)
    mod = ManifestAlloc()(mod)
    # pylint: disable=line-too-long
    # def @main(%param_0: Tensor[(5, 5), float32],
    #           %param_1: Tensor[(5, 5), float32],
    #           %param_2: Tensor[(5, 5), float32]) -> Tensor[(5, 5), float32] {
    #   let %x_0 = nullptr /* ty=() */;
    #   let %x_1 = nullptr /* ty=() */;
    #   let %x_2 = raf.op.vm.alloc_storage(int64(100), int64(64), int32(1), int32(0), str"float32");
    #   let %x_3 = raf.op.vm.alloc_tensor(%x_2, [5, 5], str"float32", [5, 5]);
    #   let %x_4 = raf.op.add;
    #   let %x_5 = (%param_0, %param_0, %x_0, %x_1);
    #   let %x_6 = (%x_3,);
    #   let %x_7 = raf.op.vm.invoke_op(%x_4, %x_5, %x_6);
    #   let %a1 = %x_3;
    #   let %x_8 = nullptr /* ty=() */;
    #   let %x_9 = nullptr /* ty=() */;
    #   let %x_10 = raf.op.vm.alloc_storage(int64(100), int64(64), int32(1), int32(0), str"float32");
    #   let %x_11 = raf.op.vm.alloc_tensor(%x_10, [5, 5], str"float32", [5, 5]);
    #   let %x_12 = raf.op.add;
    #   let %x_13 = (%a1, %param_1, %x_8, %x_9);
    #   let %x_14 = (%x_11,);
    #   let %x_15 = raf.op.vm.invoke_op(%x_12, %x_13, %x_14);
    #   let %a2 = %x_11;
    #   let %x_16 = nullptr /* ty=() */;
    #   let %x_17 = nullptr /* ty=() */;
    #   let %x_18 = raf.op.vm.alloc_storage(int64(100), int64(64), int32(1), int32(0), str"float32");
    #   let %x_19 = raf.op.vm.alloc_tensor(%x_18, [5, 5], str"float32", [5, 5]);
    #   let %x_20 = raf.op.add;
    #   let %x_21 = (%a2, %param_2, %x_16, %x_17);
    #   let %x_22 = (%x_19,);
    #   let %x_23 = raf.op.vm.invoke_op(%x_20, %x_21, %x_22);
    #   let %a3 = %x_19;
    #   %a3
    # }
    # pylint: enable=line-too-long

    expected = {
        "x_2": {-1: ["param_0", "param_1", "param_2"]},
        "x_3": {-1: ["param_0", "param_1", "param_2", "t_0"]},
        "x_4": {-1: ["param_0", "param_1", "t_1", "param_2"]},
        "x_5": {-1: ["param_0", "param_1", "t_1", "param_2"]},
        "x_6": {-1: ["param_0", "param_1", "t_1", "param_2"]},
        "x_7": {-1: ["param_0", "param_1", "t_1", "param_2"]},
        "a1": {-1: ["param_1", "t_1", "param_2"]},
        "x_10": {-1: ["param_1", "t_1", "param_2"]},
        "x_11": {-1: ["param_1", "t_1", "param_2", "t_2"]},
        "x_12": {-1: ["param_1", "t_1", "param_2", "t_3"]},
        "x_13": {-1: ["param_1", "t_1", "param_2", "t_3"]},
        "x_14": {-1: ["param_1", "t_1", "param_2", "t_3"]},
        "x_15": {-1: ["param_1", "t_1", "param_2", "t_3"]},
        "a2": {-1: ["t_3", "param_2"]},
        "x_18": {-1: ["t_3", "param_2"]},
        "x_19": {-1: ["t_3", "param_2", "t_4"]},
        "x_20": {-1: ["param_2", "t_5", "t_3"]},
        "x_21": {-1: ["param_2", "t_5", "t_3"]},
        "x_22": {-1: ["param_2", "t_5", "t_3"]},
        "x_23": {-1: ["param_2", "t_5", "t_3"]},
        "a3": {-1: ["t_5"]},
        "n_4": {-1: ["t_5"]},
        "n_3": {},
    }

    verify_live_in_set(mod, expected)


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
def test_fuse_closure():
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, p0, p1, p2):
            t_0 = raf.matmul(p0, p1)
            t_1 = raf.multiply(t_0, p2)
            t_2 = raf.relu(t_1)
            return t_2

    model = Model()
    model.infer_mode()

    device = "cpu"
    shape = (5, 5)
    m_p0, _ = randn(shape, device=device)
    m_p1, _ = randn(shape, device=device)
    m_p2, _ = randn(shape, device=device)
    args = [m_p0, m_p1, m_p2]

    mod = model._internal(*args).mod
    with raf.device("cuda"):
        mod = raf._ffi.pass_.ToGraphNormalForm()(mod)
        mod = raf._ffi.pass_.ToBasicBlockNormalForm()(mod)
        mod = raf._ffi.pass_.FuseDialect()(mod)
        mod = raf._ffi.pass_.FuseTVM()(mod)
        mod = raf._ffi.pass_.ToANormalForm()(mod)
        mod = raf._ffi.pass_.InlinePrimitives()(mod)
    # fn (%p0: Tensor[(5, 5), float32],
    #     %p1: Tensor[(5, 5), float32],
    #     %p2: Tensor[(5, 5), float32]) -> Tensor[(5, 5), float32] {
    #   let %x1 = raf.op.cublas.matmul(%p0, %p1) /* ty=Tensor[(5, 5), float32] */;
    #   %1 = fn (%p01: Tensor[(5, 5), float32], %p11: Tensor[(5, 5), float32],
    #            Primitive=1, Dialect="tvm") -> Tensor[(5, 5), float32] {
    #     %0 = raf.op.tvm.multiply(%p01, %p11);
    #     raf.op.tvm.relu(%0)
    #   };
    #   let %x3 = %1(%x1, %p2);
    #   %x3
    # }
    expected = {
        "n_0": {},
        "x1": {-1: ["param_0", "param_1", "param_2"]},
        "x3": {-1: ["param_2", "t_0"]},
        "n_1": {-1: ["t_1"]},
    }
    verify_live_in_set(mod, expected)
    mod = InferType()(mod)
    mod = ManifestAlloc()(mod)

    # def @main(%p0: Tensor[(5, 5), float32],
    #           %p1: Tensor[(5, 5), float32],
    #           %p2: Tensor[(5, 5), float32]) -> Tensor[(5, 5), float32] {
    #   let %x_0 = raf.op.vm.alloc_storage(int64(100), int64(64), int32(1), int32(0), str"float32");
    #   let %x_1 = raf.op.vm.alloc_tensor(%x_0, [5, 5], str"float32",[5, 5]);
    #   let %x_2 = raf.op.cublas.matmul;
    #   let %x_3 = (%p0, %p1);
    #   let %x_4 = (%x_1,);
    #   let %x_5 = raf.op.vm.invoke_op(%x_2, %x_3, %x_4);
    #   let %x1 = %x_1;
    #   let %x_6 = raf.op.vm.alloc_storage(int64(100), int64(64), int32(1), int32(0), str"float32");
    #   let %x_7 = raf.op.vm.alloc_tensor(%x_6, [5, 5], str"float32",[5, 5]);
    #   let %x_8 = fn (%p01: Tensor[(5, 5), float32],
    #                  %p11: Tensor[(5, 5), float32], Primitive=1, Dialect="tvm")
    #              -> Tensor[(5, 5), float32] {
    #     %0 = raf.op.tvm.add(%p01, %p11, nullptr /* ty=() */, nullptr /* ty=() */);
    #     raf.op.tvm.relu(%0)
    #   };
    #   let %x_9 = (%x1, %p2);
    #   let %x_10 = (%x_7,);
    #   let %x_11 = raf.op.vm.invoke_op(%x_8, %x_9, %x_10);
    #   let %x3 = %x_7;
    #   %x3
    # }
    expected = {
        "n_3": {},
        "x_0": {-1: ["param_0", "param_1", "param_2"]},
        "x_1": {-1: ["param_0", "param_1", "param_2", "t_0"]},
        "x_2": {-1: ["param_0", "param_1", "param_2", "t_1"]},
        "x_3": {-1: ["param_0", "param_1", "param_2", "t_1"]},
        "x_4": {-1: ["param_0", "param_1", "param_2", "t_1"]},
        "x_5": {-1: ["param_0", "param_1", "param_2", "t_1"]},
        "x1": {-1: ["param_2", "t_1"]},
        "x_6": {-1: ["param_2", "t_1"]},
        "x_7": {-1: ["param_2", "t_1", "t_2"]},
        "x_8": {-1: ["param_2", "t_1", "t_3"]},
        "x_9": {-1: ["param_2", "t_1", "t_3"]},
        "x_10": {-1: ["param_2", "t_1", "t_3"]},
        "x_11": {-1: ["param_2", "t_1", "t_3"]},
        "x3": {-1: ["t_3"]},
        "n_4": {-1: ["t_3"]},
    }
    verify_live_in_set(mod, expected)


if __name__ == "__main__":
    pytest.main([__file__])
