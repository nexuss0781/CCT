import importlib.util
import sys
import os

# Load the native Rust module
so_path = os.path.join(os.path.dirname(__file__), "_causa.so")
spec = importlib.util.spec_from_file_location("causa_native", so_path)
if spec is None:
    raise ImportError(f"Could not find shared object file at {so_path}")
causa_native = importlib.util.module_from_spec(spec)
if spec.loader is None:
    raise ImportError(f"Could not load shared object file at {so_path}")
spec.loader.exec_module(causa_native)
sys.modules.setdefault("causa_native", causa_native)


# Expose the Rust objects
Manifold = causa_native.Manifold
Event = causa_native.Event

# Expose the Python modules
from . import physics
from . import objective
from . import resonance

__all__ = [
    "Manifold",
    "Event",
    "physics",
    "objective",
    "resonance",
]