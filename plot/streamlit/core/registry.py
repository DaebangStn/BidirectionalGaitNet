"""Dynamic view module loader."""
import importlib


def load_view(module_path: str):
    """Dynamically import a view module."""
    return importlib.import_module(module_path)
