"""Dependency-only bundle requirement normalization.

This module deliberately stays free of the serving/runtime dependency graph so
release tooling can hash bundle requirements without installing Torch or CUDA.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import cast

from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version

_CUDA_ONLY_PACKAGES = frozenset({"fla-core", "flash-attn", "xformers"})
_EXACT_OPERATORS = frozenset({"==", "==="})


def _normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def resolve_bundle_requirements(
    bundle_deps: Mapping[str, object],
    *,
    exclude_cuda: bool = False,
) -> list[str]:
    """Convert a bundle ``deps`` mapping into canonical PEP 508 strings."""
    requirements: list[str] = []
    for package, constraint in bundle_deps.items():
        normalized = _normalize_package_name(package)
        if exclude_cuda and normalized in _CUDA_ONLY_PACKAGES:
            continue

        if isinstance(constraint, Mapping):
            fields = cast("Mapping[str, object]", constraint)
            url = fields.get("url", "")
            marker = fields.get("marker", "")
            version = fields.get("version", "")
            if url:
                dependency = f"{package} @ {url}"
                if marker:
                    dependency += f" ; {marker}"
                requirements.append(dependency)
            elif version:
                dependency = f"{package}{version}"
                if marker:
                    dependency += f" ; {marker}"
                requirements.append(dependency)
            continue

        requirements.append(f"{package}{constraint}" if constraint else package)
    return requirements


def normalized_bundle_requirements(bundle_deps: Mapping[str, object]) -> list[str]:
    """Return the marker-free, sorted requirements used by release pins."""
    return sorted(
        requirement.split(";", maxsplit=1)[0].strip() for requirement in resolve_bundle_requirements(bundle_deps)
    )


def bundle_requirements_sha256(bundle_deps: Mapping[str, object]) -> str:
    """Hash the exact normalized requirement payload baked by worker images."""
    payload = "\n".join(normalized_bundle_requirements(bundle_deps)).encode()
    return sha256(payload).hexdigest()


def locked_versions_from_uv_lock(source: str | Path) -> dict[str, str]:
    """Return locked constraint versions keyed by normalized package name.

    One lock entry becomes that exact version string. Several entries that
    share a public version (``torch`` ``2.9.1`` and ``2.9.1+cu129``) become a
    prefix match, ``2.9.1.*``, so each locked build matches and a newer
    release does not. Distinct public versions are an error.
    """
    text = source.read_text(encoding="utf-8") if isinstance(source, Path) else source
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        msg = f"invalid uv.lock: {exc}"
        raise ValueError(msg) from exc
    packages = data.get("package", [])
    if not isinstance(packages, list):
        msg = "uv.lock package table is not a list"
        raise ValueError(msg)

    found: dict[str, set[str]] = {}
    for package in packages:
        if not isinstance(package, dict):
            continue
        name = package.get("name")
        version = package.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            continue
        found.setdefault(_normalize_package_name(name), set()).add(version)

    return {name: _constraint_version(name, versions) for name, versions in found.items()}


def merge_locked_requirements(requirements: Sequence[str], locked_versions: Mapping[str, str]) -> list[str]:
    """Pin ranged requirements to the lock unless the bundle spec overrides it.

    Exact ``==`` / ``===`` pins and URL or VCS specs are returned unchanged.
    A range is rewritten only when a locked version still satisfies it, so a
    bundle can deliberately move off the lock (transformers 5 versus a 4.x
    pin) without an unsatisfiable constraint. Packages missing from the lock
    stay as written.
    """
    merged: list[str] = []
    for requirement in requirements:
        parsed = Requirement(requirement)
        if _is_bundle_override(parsed):
            merged.append(requirement)
            continue
        pinned = locked_versions.get(_normalize_package_name(parsed.name))
        if pinned is None or not _locked_constraint_satisfies(parsed, pinned):
            merged.append(requirement)
            continue
        merged.append(_format_locked_requirement(requirement, parsed, pinned))
    return merged


def lock_constraint_lines(requirements: Sequence[str], locked_versions: Mapping[str, str]) -> list[str]:
    """Return only the lock pins a constraints file should add."""
    merged = merge_locked_requirements(requirements, locked_versions)
    return [updated for original, updated in zip(requirements, merged, strict=True) if updated != original]


def _constraint_version(name: str, versions: set[str]) -> str:
    if len(versions) == 1:
        return next(iter(versions))
    try:
        public_versions = {Version(version).public for version in versions}
    except InvalidVersion as exc:
        msg = f"uv.lock pins {name} to an invalid version"
        raise ValueError(msg) from exc
    if len(public_versions) != 1:
        joined = ", ".join(sorted(versions))
        msg = f"uv.lock pins {name} to multiple public versions: {joined}"
        raise ValueError(msg)
    return f"{next(iter(public_versions))}.*"


def _locked_constraint_satisfies(requirement: Requirement, constraint: str) -> bool:
    """A prefix pin is usable when its public version is inside the bundle spec."""
    candidate = constraint.removesuffix(".*") if constraint.endswith(".*") else constraint
    try:
        version = Version(candidate)
    except InvalidVersion:
        return False
    return requirement.specifier.contains(version, prereleases=True)


def _is_bundle_override(requirement: Requirement) -> bool:
    """Exact pins and direct URL/VCS references are deliberate bundle overrides."""
    if requirement.url is not None:
        return True
    specs = list(requirement.specifier)
    if len(specs) != 1:
        return False
    spec = specs[0]
    return spec.operator in _EXACT_OPERATORS and not spec.version.endswith(".*")


def _format_locked_requirement(raw: str, requirement: Requirement, version: str) -> str:
    extras = f"[{','.join(sorted(requirement.extras))}]" if requirement.extras else ""
    pinned = f"{requirement.name}{extras}=={version}"
    if requirement.marker is None:
        return pinned
    marker = raw.split(";", maxsplit=1)[1].strip()
    return f"{pinned} ; {marker}"
