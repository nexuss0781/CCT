import importlib.util
import sys
import os

# Load Rust shared lib named 'causa_native.so' (not causa_py.so)
so_path = os.path.join(os.path.dirname(__file__), "causa_native.so")

spec = importlib.util.spec_from_file_location("causa_native", so_path)
causa_native = importlib.util.module_from_spec(spec)
sys.modules["causa_native"] = causa_native
spec.loader.exec_module(causa_native)

# Expose Rust classes at package level
Manifold = causa_native.Manifold
Event = causa_native.Event

# Import Python submodule 'physics.py' from the same package
from . import physics