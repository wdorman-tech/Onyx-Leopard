from __future__ import annotations

from collections.abc import Callable

from packaging.version import Version


def migrate_0_9_to_1_0(data: dict) -> dict:
    """Example migration from 0.9.0 to 1.0.0."""
    # Add schema_version to profile if missing
    if "profile" in data and "schema_version" not in data["profile"]:
        data["profile"]["schema_version"] = "1.0.0"

    # Add sim_params section if missing
    if "profile" in data and "sim_params" not in data["profile"]:
        data["profile"]["sim_params"] = {}

    # Add metadata section if missing
    if "profile" in data and "metadata" not in data["profile"]:
        data["profile"]["metadata"] = {}

    data["format_version"] = "1.0.0"
    return data


# Ordered migration chain: (from_version, to_version, function)
MIGRATIONS: list[tuple[str, str, Callable[[dict], dict]]] = [
    ("0.9.0", "1.0.0", migrate_0_9_to_1_0),
]

SUPPORTED_VERSIONS = {"1.0.0", "0.9.0"}
CURRENT_VERSION = "1.0.0"


def needs_migration(version: str) -> bool:
    return version != CURRENT_VERSION


def migrate(data: dict) -> dict:
    """Apply migration chain to bring data to current version."""
    current = data.get("format_version", "1.0.0")

    if current == CURRENT_VERSION:
        return data

    for from_v, to_v, fn in MIGRATIONS:
        if Version(current) == Version(from_v):
            data = fn(data)
            current = to_v

    return data
