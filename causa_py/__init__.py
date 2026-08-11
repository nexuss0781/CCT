"""Public Python interface for the CCT prototype.

The native Rust extension is deliberately imported through its canonical
module name. This avoids loading stale shared objects from the source tree and
makes a missing native build an explicit, actionable installation error.
"""

try:
    from causa_native import Event, Manifold
except ImportError as exc:  # pragma: no cover - exercised in clean-install diagnostics
    raise ImportError(
        "The CCT native extension is unavailable. Build it with "
        "`make install-native` (or `maturin develop --manifest-path "
        "causa_core/Cargo.toml`)."
    ) from exc

from . import physics

__all__ = ["Event", "Manifold", "physics"]
