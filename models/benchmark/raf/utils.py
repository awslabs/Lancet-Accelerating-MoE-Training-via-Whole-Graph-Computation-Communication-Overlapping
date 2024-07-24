"""Utilities."""
# pylint: disable=not-callable

def patch_deepspeed_moe_conversion_map(gate_type='switch'):
    # register additional conversion rules
    # register scatter_ conversion
    import tvm
    from tvm import relay

    def make_scatter_(inputs, input_types):
        data = inputs[0]
        axis = int(inputs[1])
        index = inputs[2]
        src = relay.multiply(relay.ones_like(data), relay.cast_like(relay.const(inputs[3]), data))
        return relay.op.transform.scatter(data, index, src, axis)
    
    # NOTE: this convert any primitive python op. should only be used
    # for our custom moe layer (and make sure no other parts of the model has prim::PythonOp)
    def make_moe(inputs, input_types):
        if len(inputs) == 1:
            # all to all
            data = inputs[0]
            data = relay.Tuple([data])
            return relay.Call(relay.op.get("raf.op._all_to_all"), [data])
        if len(inputs) == 4:
            # moe encode
            data = inputs[0]
            gate = inputs[1]
            used_capacity = inputs[2]
            # inputs[3] is unused
            # output is [gate_s, indices_locations, used_capacity, n_elements_per_expert, dispatched_input]
            if gate_type == "switch":
                encode_output = relay.Call(relay.op.get("raf.op.moe_encode"), [data, gate, used_capacity])
            else:
                assert gate_type == "batch_prioritized", "gate_type must be either switch or batch_prioritized"
                encode_output = relay.Call(relay.op.get("raf.op.moe_encode_batch_prioritized"), [data, gate])
            return (relay.TupleGetItem(encode_output, 0),
                    relay.TupleGetItem(encode_output, 1),
                    relay.TupleGetItem(encode_output, 2),
                    relay.TupleGetItem(encode_output, 3),
                    relay.TupleGetItem(encode_output, 4))
        else:
            assert len(inputs) == 3
            # moe decode
            data, gate, indices_locations = inputs
            return relay.Call(relay.op.get("raf.op.moe_decode"), [data, gate, indices_locations])

    custom_convert_map = {"aten::scatter_": make_scatter_, "prim::PythonOp": make_moe}

    # also monkeypatch PyTorchOpConverter (in tvm.relay.frontend.pytorch)
    from tvm.relay.frontend.pytorch import PyTorchOpConverter

    def patched_to(self, inputs, input_types):
        data = inputs[0]
        dtype = inputs[1] if inputs[1] is not None and not isinstance(inputs[1], str) else inputs[2]
        # special handling for aten::to(data, 6, _, _, _) case
        # 6 means dtype = float
        # this happens when converting upsampling with scale factor

        # NOTE(monkeypatch): also handle 11 (dtype = bool)
        cast_map = {
            5: "float16",
            6: "float32",
            7: "float64",
            3: "int32",
            4: "int64",
            11: "bool"
        }

        cast_func = {5: float, 6: float, 7: float, 3: int, 4: int, 11: bool}

        ret = data
        if isinstance(data, relay.Expr):
            actual_dtype = str(self.infer_type(data).dtype)
            if dtype in cast_map and cast_map[dtype] != actual_dtype:
                ret = relay.op.cast(data, cast_map[dtype])
        elif dtype in cast_map:
            ret = cast_func[dtype](data)

        return ret
    PyTorchOpConverter.to = patched_to
    return custom_convert_map
