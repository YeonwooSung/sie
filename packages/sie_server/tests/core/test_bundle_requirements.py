from __future__ import annotations

from pathlib import Path

import yaml
from sie_server.bundle_requirements import (
    bundle_requirements_sha256,
    lock_constraint_lines,
    locked_versions_from_uv_lock,
    merge_locked_requirements,
    normalized_bundle_requirements,
    resolve_bundle_requirements,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_BUNDLES = _REPO_ROOT / "packages" / "sie_server" / "bundles"


def test_resolve_bundle_requirements_preserves_cli_semantics() -> None:
    assert resolve_bundle_requirements(
        {
            "plain": "",
            "bounded": ">=1,<2",
            "wheel": {"url": "https://example.com/wheel.whl", "marker": "sys_platform == 'linux'"},
            "versioned": {"version": "==3", "marker": "sys_platform == 'darwin'"},
        }
    ) == [
        "plain",
        "bounded>=1,<2",
        "wheel @ https://example.com/wheel.whl ; sys_platform == 'linux'",
        "versioned==3 ; sys_platform == 'darwin'",
    ]


def test_resolve_bundle_requirements_can_exclude_normalized_cuda_packages() -> None:
    assert resolve_bundle_requirements(
        {"Flash_Attn": "==1", "xformers": "==2", "FLA_core": {"version": "==3"}, "portable": "==4"},
        exclude_cuda=True,
    ) == ["portable==4"]


def test_release_pin_normalization_is_sorted_and_marker_free() -> None:
    deps = {
        "z-last": "==2",
        "a-first": {"version": "==1", "marker": "sys_platform == 'linux'"},
    }

    assert normalized_bundle_requirements(deps) == ["a-first==1", "z-last==2"]
    assert bundle_requirements_sha256(deps) == "d2ccdab8db5e0df55e2b47ae7c7a8f4c9b7623bd42fb7c09f028114ed3aba884"


def _bundle_requirements(name: str) -> list[str]:
    bundle = yaml.safe_load((_BUNDLES / f"{name}.yaml").read_text())
    return resolve_bundle_requirements(bundle["deps"])


def test_default_bundle_ranges_pin_to_uv_lock_versions() -> None:
    locked = locked_versions_from_uv_lock(_REPO_ROOT / "uv.lock")
    original = _bundle_requirements("default")
    merged = merge_locked_requirements(original, locked)

    assert "gliner>=0.2.26,<1" in original
    assert f"gliner=={locked['gliner']}" in merged
    assert locked["gliner"] == "0.2.26"
    assert "gliner>=0.2.26,<1" not in merged
    assert f"torch=={locked['torch']}" in merged
    assert f"requests=={locked['requests']}" in merged


def test_exact_pins_and_url_specs_override_the_lock() -> None:
    locked = locked_versions_from_uv_lock(_REPO_ROOT / "uv.lock")
    original = _bundle_requirements("transformers5")
    merged = merge_locked_requirements(original, locked)
    url_specs = [requirement for requirement in original if " @" in requirement]

    assert url_specs
    assert locked["gliner2"] == "1.3.2"
    assert "gliner2==2.0.0" in merged
    assert "torchvision==0.24.1" in merged
    assert locked["torchvision"] != "0.24.1"
    for requirement in url_specs:
        assert requirement in merged
    # Ranges that do not contain the locked version stay as written.
    assert "transformers>=5.14,<6" in merged
    assert "sentence-transformers>=5.6,<6" in merged
    constraints = lock_constraint_lines(original, locked)
    assert "gliner2==2.0.0" not in constraints
    assert "torchvision==0.24.1" not in constraints
    assert all(" @" not in line for line in constraints)
    assert "transformers==4.57.6" not in constraints


def test_merge_leaves_url_specs_and_unlocked_packages_unchanged() -> None:
    locked = {"gliner": "0.2.26", "demo": "9.9.9"}
    requirements = [
        "gliner>=0.2.26,<1 ; sys_platform == 'linux'",
        "demo @ git+https://example.com/demo.git@abc123",
        "wheel @ https://example.com/wheel.whl ; sys_platform == 'linux'",
        "absent>=1,<2",
        "pinned==3",
    ]

    assert merge_locked_requirements(requirements, locked) == [
        "gliner==0.2.26 ; sys_platform == 'linux'",
        "demo @ git+https://example.com/demo.git@abc123",
        "wheel @ https://example.com/wheel.whl ; sys_platform == 'linux'",
        "absent>=1,<2",
        "pinned==3",
    ]


def test_local_version_builds_share_one_prefix_constraint() -> None:
    lock = """
version = 1
[[package]]
name = "torch"
version = "2.9.1"

[[package]]
name = "torch"
version = "2.9.1+cu129"
"""
    assert locked_versions_from_uv_lock(lock) == {"torch": "2.9.1.*"}
    assert merge_locked_requirements(["torch>=2.9,<2.10"], {"torch": "2.9.1.*"}) == ["torch==2.9.1.*"]
