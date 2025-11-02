import importlib


import importlib

def get_filter(filter_name: str = "blur"):
    """load a function from a module in the images package"""
    module = importlib.import_module(f"images.{filter_name}")
    return getattr(module, filter_name)
