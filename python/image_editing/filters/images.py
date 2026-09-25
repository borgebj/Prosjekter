import importlib
import sys
from pathlib import Path


def get_filter(filter: str = "blur", implementation: str = "python"):
    """Load a filter function from a module.

    Assumes filters are named e.g. greyscale.numpy_greyscale.

    Args:
        filter (str):
            The name of the filter (e.g. "greyscale")
        implementation (str):
            The implementation (python, numpy)

    Returns:
        filter_function (function)
    """
    folder = Path(__file__).resolve().parent
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

    # Get the module, e.g. filters.greyscale
    module = importlib.import_module(filter)

    # Construct filter function name, e.g. numpy_greyscale
    filter_name = f"{implementation}_{filter}"

    # Return the function from the module
    return getattr(module, filter_name)
