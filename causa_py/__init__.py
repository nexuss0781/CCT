# === /content/CCT/causa_py/__init__.py ===

# Load Rust bindings from causa_native.so
import importlib.util
import sys
import os

_native_path = os.path.join(os.path.dirname(__file__), "causa_native.so")
spec = importlib.util.spec_from_file_location("causa_native", _native_path)
causa_native = importlib.util.module_from_spec(spec)
sys.modules["causa_native"] = causa_native
spec.loader.exec_module(causa_native)

Manifold = causa_native.Manifold
Event = causa_native.Event

# ✅ Only after loading native, import local Python files
import causa_py.physics  # NOT "from . import physics"