"""Build portably, or use the verified frozen profile on a native CM5.

profiles/cm5.profdata is copied unchanged from
native/eagle_tags/results/rust-migration-v1/frozen-rust-combined-pgo/profile.profdata;
cm5.json retains only its compiler, profile identity, and frozen source hashes.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tomllib
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
MODEL_PATHS = (
    Path("/proc/device-tree/model"),
    Path("/sys/firmware/devicetree/base/model"),
)


def is_cm5() -> bool:
    """Require Linux/aarch64 and the actual Compute Module 5 device-tree name."""
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        return False
    for path in MODEL_PATHS:
        try:
            model = path.read_bytes().rstrip(b"\0").decode("ascii").strip()
        except (OSError, UnicodeError):
            continue
        if re.fullmatch(r"Raspberry Pi Compute Module 5(?: Rev \d+\.\d+)?", model):
            return True
    return False


def compiler_identity(env: dict[str, str], module_dir: Path) -> str:
    """Query the effective rustup compiler in the module's working directory."""
    try:
        result = subprocess.run(
            [env.get("RUSTC", "rustc"), "-vV"],
            cwd=module_dir,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    except OSError:
        return "unavailable"


def cargo_configuration(module_dir: Path, env: dict[str, str]) -> dict[str, str]:
    """Read applicable Cargo configs so explicit target/flag settings win."""
    cargo_dirs = [
        directory / ".cargo" for directory in (module_dir, *module_dir.parents)
    ]
    cargo_dirs.append(Path(env.get("CARGO_HOME", str(Path.home() / ".cargo"))))
    configs = {}
    for cargo_dir in cargo_dirs:
        for name in ("config", "config.toml"):
            path = cargo_dir / name
            if path.is_file():
                configs[str(path)] = path.read_text()
    return configs


def build_context(module_dir: Path, env: dict[str, str]) -> dict:
    """Resolve the build decision and cache inputs without changing the caller env."""
    env = env.copy()
    compiler = compiler_identity(env, module_dir)
    configs = cargo_configuration(module_dir, env)
    explicit = any(
        key in env
        for key in (
            "RUSTFLAGS",
            "RUSTC",
            "CARGO_ENCODED_RUSTFLAGS",
            "CARGO_BUILD_TARGET",
            "CARGO_BUILD_RUSTFLAGS",
            "CARGO_BUILD_RUSTC",
            "RUSTC_WRAPPER",
            "RUSTC_WORKSPACE_WRAPPER",
            "CARGO_BUILD_RUSTC_WRAPPER",
            "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER",
        )
    ) or any(
        key.startswith("CARGO_TARGET_") and key.endswith("_RUSTFLAGS") for key in env
    )
    for text in configs.values():
        # Invalid config will fail in Cargo; never guess a tuned build in that case.
        try:
            config = tomllib.loads(text)
            explicit |= any(
                key in config.get("build", {})
                for key in (
                    "target",
                    "rustflags",
                    "rustc",
                    "rustc-wrapper",
                    "rustc-workspace-wrapper",
                )
            )
            explicit |= any(
                "rustflags" in settings
                for settings in config.get("target", {}).values()
            )
        except (tomllib.TOMLDecodeError, TypeError, AttributeError):
            explicit = True
    cm5 = is_cm5()
    reason = "Portable build (not a native Linux/aarch64 Compute Module 5)."
    if cm5:
        env.setdefault("CARGO_BUILD_JOBS", "2")
        if explicit:
            reason = "Automatic CM5 tuning disabled: respecting explicit Cargo target/Rust flags/compiler overrides."
        elif any(
            line.startswith("host: ") and line != "host: aarch64-unknown-linux-gnu"
            for line in compiler.splitlines()
        ):
            reason = "Portable build: compiler host is not native aarch64 Linux."
        else:
            env["CARGO_ENCODED_RUSTFLAGS"] = "-C\x1ftarget-cpu=cortex-a76"
            reason = "CM5 Cortex-A76 tuning; frozen PGO unavailable."
            try:
                manifest = json.loads(
                    (module_dir / "profiles" / "cm5.json").read_text()
                )
                profile = module_dir / "profiles" / "cm5.profdata"
                compiler_ok = (
                    compiler.splitlines()[0] == manifest["rustc"]
                    and (f"LLVM version: {manifest['llvm']}" in compiler.splitlines())
                    and "host: aarch64-unknown-linux-gnu" in compiler.splitlines()
                )
                expected = manifest["source_hashes"]
                actual = {
                    str(path.relative_to(module_dir)): hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    for path in sorted(
                        [*module_dir.glob("*.rs"), *(module_dir / "src").rglob("*.rs")]
                    )
                }
                actual.update(
                    {
                        name: hashlib.sha256(
                            (module_dir / name).read_bytes()
                        ).hexdigest()
                        for name in expected
                        if not name.endswith(".rs")
                    }
                )
                source_ok = actual == expected
                profile_ok = (
                    profile.stat().st_size == manifest["profile_bytes"]
                    and hashlib.sha256(profile.read_bytes()).hexdigest()
                    == manifest["profile_sha256"]
                )
                if compiler_ok and source_ok and profile_ok:
                    env["CARGO_ENCODED_RUSTFLAGS"] += (
                        f"\x1f-C\x1fprofile-use={profile.resolve()}"
                    )
                    reason = "CM5 Cortex-A76 tuning with verified frozen PGO."
                else:
                    mismatches = [
                        name
                        for name, ok in (
                            ("compiler", compiler_ok),
                            ("source snapshot", source_ok),
                            ("profile checksum", profile_ok),
                        )
                        if not ok
                    ]
                    reason += " Mismatch: " + ", ".join(mismatches) + "."
            except (OSError, ValueError, KeyError, IndexError):
                reason += " Missing or invalid profile metadata/data."
            if "profile-use=" not in env["CARGO_ENCODED_RUSTFLAGS"]:
                reason += (
                    " Recorded performance is not guaranteed. Use the pinned rustc/LLVM and source snapshot "
                    "in profiles/cm5.json, or explicitly validate a newly trained profile; stale PGO was not used."
                )
    return {
        "env": env,
        "compiler": compiler,
        "configs": configs,
        "cm5": cm5,
        "reason": reason,
    }


def main() -> int:
    """Expose cache decisions or install using the same resolved build environment."""
    context = build_context(MODULE_DIR, dict(os.environ))
    if sys.argv[1:] == ["--build-context"]:
        # Only relevant build env belongs in the cache, not credentials or volatile shell state.
        context["env"] = {
            key: value
            for key, value in context["env"].items()
            if key
            in {
                "RUSTFLAGS",
                "CARGO_ENCODED_RUSTFLAGS",
                "CARGO_BUILD_TARGET",
                "CARGO_BUILD_RUSTFLAGS",
                "CARGO_BUILD_RUSTC",
                "RUSTC",
                "RUSTUP_TOOLCHAIN",
                "CARGO_HOME",
                "CARGO_BUILD_JOBS",
                "RUSTC_WRAPPER",
                "RUSTC_WORKSPACE_WRAPPER",
                "CARGO_BUILD_RUSTC_WRAPPER",
                "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER",
            }
            or (key.startswith("CARGO_TARGET_") and key.endswith("_RUSTFLAGS"))
        }
        context["configs"] = {
            path: hashlib.sha256(text.encode()).hexdigest()
            for path, text in context["configs"].items()
        }
        print(json.dumps(context, sort_keys=True))
        return 0
    if sys.argv[1:]:
        print("Usage: build.py [--build-context]", file=sys.stderr)
        return 2
    print(context["reason"], flush=True)
    return subprocess.run(
        [sys.executable, "-m", "maturin", "develop", "--release", "--locked"],
        cwd=MODULE_DIR,
        env=context["env"],
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
