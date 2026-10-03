"""Fail-closed role checks for offline fitting, never production routing.

Raw numerical arrays have no provenance. Fitting requires an explicit development
declaration or role-bound labels obtained from a verified development manifest.
An explicit declaration cannot override a locked-test marker. This prevents
accidental use through standard tooling, not deliberate relabelling by an owner.
"""
from __future__ import annotations

import numpy as np

ROLES = {"train": "training", "training": "training", "dev": "development",
         "development": "development", "test": "locked_test", "locked_test": "locked_test"}


class RoleLabels(np.ndarray):
    def __new__(cls, values, role: str):
        obj = np.array(values, dtype=np.int64, copy=True).view(cls)
        obj.data_role = ROLES.get(role, role)
        obj.setflags(write=False)
        return obj

    def __array_finalize__(self, source):
        self.data_role = getattr(source, "data_role", None)


def require_development(labels, declared_role: str | None = None) -> None:
    bound = getattr(labels, "data_role", None)
    if bound is not None and ROLES.get(bound, bound) != "development":
        raise ValueError("calibration/threshold selection requires development data; locked test is forbidden")
    role = ROLES.get(declared_role, declared_role) if declared_role is not None else bound
    if role != "development":
        raise ValueError("calibration requires explicit development provenance; unmarked arrays are refused")


def require_rows_role(rows: list[dict], expected: str) -> None:
    expected = ROLES.get(expected, expected)
    for row in rows:
        marker = row.get("data_role")
        if marker is not None and ROLES.get(marker, marker) != expected:
            raise ValueError(f"{expected} input contains {marker} rows; locked test cannot be fitted")
