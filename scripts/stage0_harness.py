#!/usr/bin/env python3
"""Stage 0 reproducibility and gate harness.

The harness intentionally uses a small deterministic workload. It is not a
performance benchmark for future CCT-ASE models; it verifies that the package,
native extension, baseline numerical path, metadata schema, and gate evaluator
are functioning and auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "benchmark_record.schema.json"


@dataclass
class CheckRecord:
    name: str
    status: str
    duration_seconds: float
    details: dict[str, Any]
    required: bool = True


@dataclass
class BenchmarkRecord:
    name: str
    value: float
    unit: str
    seed: int
    commit: str
    config_hash: str
    hardware: str
    timestamp_utc: str
    status: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_value(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def environment_manifest() -> dict[str, Any]:
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "commit": git_value("rev-parse", "HEAD"),
        "dirty_tree": bool(git_value("status", "--porcelain")),
        "jax_enable_x64": os.environ.get("JAX_ENABLE_X64", "unset"),
        "timestamp_utc": utc_now(),
    }


def config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_check(name: str, function, required: bool = True) -> CheckRecord:
    started = time.perf_counter()
    try:
        details = function()
        return CheckRecord(
            name=name,
            status="PASS",
            duration_seconds=time.perf_counter() - started,
            details=details or {},
            required=required,
        )
    except Exception as exc:  # noqa: BLE001 - the harness must report failures
        return CheckRecord(
            name=name,
            status="FAIL",
            duration_seconds=time.perf_counter() - started,
            details={"error_type": type(exc).__name__, "error": str(exc)},
            required=required,
        )


def check_package_import() -> dict[str, Any]:
    import causa_native
    import causa_py

    return {
        "package": causa_py.__name__,
        "native_module": causa_native.__name__,
        "exports": sorted(name for name in ("Event", "Manifold", "physics") if hasattr(causa_py, name)),
    }


def check_event_lifecycle() -> dict[str, Any]:
    from causa_py import Event, Manifold

    manifold = Manifold([8, 8])
    event = Event([0.1, 0.2], [3, 4], [1.0, 0.5])
    manifold.place_event(event)
    retrieved = manifold.get_event([3, 4])
    if retrieved is None:
        raise AssertionError("Inserted event was not retrievable")
    if manifold.filled_cells() != 1:
        raise AssertionError("filled_cells did not report one event")
    if len(manifold.events()) != 1:
        raise AssertionError("events() did not return one event")
    return {"filled_cells": manifold.filled_cells(), "repr": repr(manifold)}


def check_invalid_inputs() -> dict[str, Any]:
    from causa_py import Event, Manifold

    errors: list[str] = []
    for dimensions in ([], [0, 4]):
        try:
            Manifold(dimensions)
        except ValueError:
            errors.append("invalid_dimensions")
    manifold = Manifold([4, 4])
    try:
        manifold.place_event(Event([], [-1, 0], []))
    except ValueError:
        errors.append("negative_coordinate")
    try:
        manifold.get_event([1])
    except ValueError:
        errors.append("wrong_dimensionality")
    event = Event([], [1, 1], [])
    manifold.place_event(event)
    try:
        manifold.place_event(event)
    except ValueError:
        errors.append("duplicate_cell")
    if len(errors) != 5:
        raise AssertionError(f"Expected five structured input errors, got {errors}")
    return {"structured_error_cases": errors}


def check_deterministic_numerical_path() -> dict[str, Any]:
    import numpy as np
    import jax.numpy as jnp
    from causa_py import Manifold
    from causa_py.physics import create_propagation_kernel, create_source_field, resolve_system

    manifold = Manifold([16, 16])
    from causa_py import Event

    manifold.place_event(Event([1.0], [8, 8], [1.0]))
    source = create_source_field(manifold)
    kernel = create_propagation_kernel((16, 16), {"decay_rate": 0.1})
    output_a = np.asarray(resolve_system(source, kernel))
    output_b = np.asarray(resolve_system(source, kernel))
    if not np.array_equal(output_a, output_b):
        raise AssertionError("Repeated baseline numerical calls were not identical")
    if not np.isfinite(output_a).all():
        raise AssertionError("Baseline response contains non-finite values")
    if float(np.max(output_a)) <= 0.0:
        raise AssertionError("Baseline response is not positive")
    digest = hashlib.sha256(output_a.tobytes()).hexdigest()
    return {"output_shape": list(output_a.shape), "output_sha256": digest, "max_value": float(np.max(output_a))}


def check_benchmark_schema() -> dict[str, Any]:
    import jsonschema

    schema = json.loads(SCHEMA_PATH.read_text())
    config = {"stage": 0, "mode": "smoke", "seed": 0}
    record = asdict(
        BenchmarkRecord(
            name="stage0_schema_smoke",
            value=1.0,
            unit="boolean",
            seed=0,
            commit=git_value("rev-parse", "HEAD"),
            config_hash=config_hash(config),
            hardware=platform.platform(),
            timestamp_utc=utc_now(),
            status="PASS",
        )
    )
    jsonschema.validate(record, schema)
    return {"schema": str(SCHEMA_PATH.relative_to(ROOT)), "record_valid": True}


def build_benchmark_records(checks: list[CheckRecord], config: dict[str, Any]) -> list[dict[str, Any]]:
    commit = git_value("rev-parse", "HEAD")
    digest = config_hash(config)
    records = []
    for check in checks:
        records.append(
            asdict(
                BenchmarkRecord(
                    name=check.name,
                    value=1.0 if check.status == "PASS" else 0.0,
                    unit="boolean",
                    seed=int(config["seed"]),
                    commit=commit,
                    config_hash=digest,
                    hardware=platform.platform(),
                    timestamp_utc=utc_now(),
                    status=check.status,
                )
            )
        )
    return records


def gate_status(checks: list[CheckRecord]) -> str:
    required = [check for check in checks if check.required]
    if any(check.status == "FAIL" for check in required):
        return "FAIL"
    if any(check.status == "BLOCKED" for check in required):
        return "BLOCKED"
    return "PASS"


def write_report(output: Path, mode: str, config: dict[str, Any], checks: list[CheckRecord]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    environment = environment_manifest()
    records = build_benchmark_records(checks, config)
    status = gate_status(checks)
    manifest = {
        "stage": 0,
        "mode": mode,
        "config": config,
        "config_hash": config_hash(config),
        "environment": environment,
        "created_at_utc": utc_now(),
        "required_checks": len([check for check in checks if check.required]),
        "passed_checks": len([check for check in checks if check.status == "PASS"]),
        "failed_checks": len([check for check in checks if check.status == "FAIL"]),
    }
    gate = {
        "stage": 0,
        "status": status,
        "transition": "Stage 1" if status == "PASS" else "STOP",
        "commit": environment["commit"],
        "config_hash": config_hash(config),
        "checks": [asdict(check) for check in checks],
        "created_at_utc": utc_now(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")
    (output / "tests.json").write_text(json.dumps([asdict(check) for check in checks], indent=2, sort_keys=True) + "\n")
    (output / "benchmarks.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    (output / "gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Stage 0 Gate Report",
        "",
        f"**Status:** `{status}`",
        f"**Transition:** `{gate['transition']}`",
        f"**Commit:** `{environment['commit']}`",
        f"**Configuration hash:** `{gate['config_hash']}`",
        "",
        "| Check | Status | Duration (s) |",
        "|---|---:|---:|",
    ]
    lines.extend(
        f"| {check.name} | `{check.status}` | {check.duration_seconds:.6f} |"
        for check in checks
    )
    lines.extend(
        [
            "",
            "A `PASS` authorizes work to prepare Stage 1 but does not authorize Stage 1 implementation without explicit user approval.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "gate"), default="smoke")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "stage-0" / "run")
    args = parser.parse_args()

    config = {"stage": 0, "mode": args.mode, "seed": 0, "workload": "baseline-16x16"}
    checks = [
        run_check("package_import", check_package_import),
        run_check("event_lifecycle", check_event_lifecycle),
        run_check("invalid_input_errors", check_invalid_inputs),
        run_check("deterministic_numerical_path", check_deterministic_numerical_path),
        run_check("benchmark_schema", check_benchmark_schema),
    ]
    write_report(args.output, args.mode, config, checks)
    print(json.dumps({"status": gate_status(checks), "output": str(args.output)}, sort_keys=True))
    return 0 if gate_status(checks) == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
