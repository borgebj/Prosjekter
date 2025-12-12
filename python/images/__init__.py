import importlib


import importlib

def get_filter(filter: str = "blur", implementation: str = "python"):
    """load a function from a module in the images package

    Assumes filters are named e.g. images.blur.python_blur

    Args:
            filter (str):
                The name of the filter (e.g. "blur")
            Implementation (str):
                The name of implementation (python, numpy)

    Returns:
            filter_function (function)
    """
    # gets the module
    module = importlib.import_module(f"images.{filter}")

    # construct filter function name
    filter_name = f"{implementation}_{filter}"

    # return resolved function (images.blur.python_blur)
    return getattr(module, filter_name)
