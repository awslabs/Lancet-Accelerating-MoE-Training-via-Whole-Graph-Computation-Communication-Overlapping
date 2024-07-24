"""
The registry of models and optimizations.
"""
# pylint: disable=too-many-arguments
import functools
import random

import numpy as np

from .utils import func_timer

# Mapping from framework to model name to its implementation.
MODEL_REGISTRY = {}


def with_seed(seed=0):
    """
    A decorator for test functions that manages rng seeds.

    Parameters
    ----------
    seed: int
        The seed to pass to np.random and random.
    """

    def test_helper(orig_test):
        @functools.wraps(orig_test)
        def test_new(*args, **kwargs):
            if seed is not None:
                this_test_seed = seed
            else:
                this_test_seed = np.random.randint(0, np.iinfo(np.int32).max)
            post_test_state = np.random.get_state()
            np.random.seed(this_test_seed)
            random.seed(this_test_seed)
            try:
                ret = orig_test(*args, **kwargs)
            except Exception as err:
                raise RuntimeError(str(err))
            finally:
                np.random.set_state(post_test_state)
            return ret

        return test_new

    return test_helper


def reg_model(framework):
    """The decorator to register a model.

    Parameters
    ----------
    framework: str
        The framework of this model.
    """

    @with_seed(0)
    def _do_reg(func):
        """Register the model.

        Parameters
        ----------
        func: Callable[[int, Optional[Any], bool], BenchModelBase]
            The function accepting batch size, input shape, and whether to convert
            the model to RAF. It then returns a BenchModel that can be used to bemchmark
            the performance.
        """
        if framework not in MODEL_REGISTRY:
            MODEL_REGISTRY[framework] = {}

        name = func.func_name if hasattr(func, "func_name") else func.__qualname__
        assert name not in MODEL_REGISTRY[framework], "%s.%s has been registered" % (
            framework,
            name,
        )
        MODEL_REGISTRY[framework][name] = func
        return func

    return _do_reg


def list_model_names(framework):
    """List all available model names.

    Parameters
    ----------
    framework: str
        The framework to be listed.

    Returns
    -------
    names: List[str]
        A list of model names.
    """
    return list(MODEL_REGISTRY[framework].keys())


@func_timer("Initialize model bencher")
def get_model_bencher(
    framework, name, batch_size=32, shape=None, dtype="float32", include_orig_model=False, **kwargs
):
    """Get the benchmark model by given its name.

    Parameters
    ----------
    framework: str
        The target framework.

    name: str
        The model name.

    batch_size: int
        The batch size. Default 32.

    shape: Optional[Any]
        The image size or sequence length. If not present, use the default defined in the model.

    dtype: str
        The data type. Default is float32.

    include_orig_model: bool
        Whether to include the original model as the reference. Default False.
        Note that the original model may also occupy device memory even not running,
        so in the case of benchamrking RAF models only, we better do not include it.

    Returns
    -------
    ModelBencherBase
        The benchmark model.
    """
    assert framework in MODEL_REGISTRY, "Unregistered framework %s" % framework
    assert name in MODEL_REGISTRY[framework], "Unregisted model %s in %s. Available: %s" % (
        name,
        framework,
        list_model_names(framework),
    )
    assert dtype in ["float32", "float16"], "Only support dtype float32 and float16"
    return MODEL_REGISTRY[framework][name](batch_size, shape, dtype, include_orig_model, **kwargs)


def get_opt(name):
    """Get the RAF optimization function.

    Parameters
    ----------
    name: str
        The optimization name.

    Returns
    -------
    func: Callable
        The optimization function.
    """
    assert name in OPT_REGISTRY, "Unregisted optimization %s. Available: %s" % (
        name,
        list_opt_names(),
    )
    return OPT_REGISTRY[name]
