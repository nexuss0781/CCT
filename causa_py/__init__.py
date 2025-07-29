import importlib.util
import sys
import os

so_path = os.path.join(os.path.dirname(__file__), "causa_py.so")

spec = importlib.util.spec_from_file_location("causa_py", so_path)
causa_native = importlib.util.module_from_spec(spec)
sys.modules["causa_py"] = causa_native
spec.loader.exec_module(causa_native)

Manifold = causa_native.Manifold
Event = causa_native.Event

from . import physics