import importlib.util
import sys
import os

so_path = os.path.join(os.path.dirname(__file__), "causa_py.so")

spec = importlib.util.spec_from_file_location("causa_py", so_path)
causa_py = importlib.util.module_from_spec(spec)
sys.modules["causa_py"] = causa_py
spec.loader.exec_module(causa_py)

Manifold = causa_py.Manifold
Event = causa_py.Event
from . import physics