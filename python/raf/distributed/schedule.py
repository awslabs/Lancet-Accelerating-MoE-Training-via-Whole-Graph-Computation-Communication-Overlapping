# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods
"""Joint Schedule Simulator"""
from tvm import relay
import raf._ffi.distributed as _ffi
from raf._core.core_utils import register_node
from raf._lib import Object

@register_node("raf.distributed.LancetScheduleSimulator")
class LancetScheduleSimulator(Object):
    def __init__(self):
        self.__init_handle_by_constructor__(_ffi.LancetScheduleSimulator)
        self.has_profile_data = False

    def load_profile(self, fn_prefix: str):
        # TODO: do some checking
        return _ffi.LancetScheduleSimulatorLoadProfile(self, fn_prefix)

    def load_ir(self, fn_path: str):
        return _ffi.LancetScheduleSimulatorLoadIR(self, fn_path)

    def run(self, expr: relay.Expr, sched_heuristic: str, timeline_opt_algo: str, dp_group_size: int):
        # TODO: check heuristic
        scheduled_expr = _ffi.LancetScheduleSimulatorRunSchedule(self, expr, sched_heuristic, timeline_opt_algo, dp_group_size)
        self.has_profile_data = True
        return scheduled_expr

    def get_profile_data(self):
        if not self.has_profile_data:
            raise RuntimeError("Should call run first to generate profile data.")
        profile_data = _ffi.LancetScheduleSimulatorGetProfileData(self)
        python_dict = {}
        for key in profile_data.keys():
            python_pairs = []
            for pair in profile_data[key]:
                python_pairs.append((pair[0].value, pair[1].value))
            python_dict[key.name] = python_pairs
        return python_dict



