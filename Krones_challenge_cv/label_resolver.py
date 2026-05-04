"""
src/utils/label_resolver.py
────────────────────────────
Converts raw dataset labels (with optional area_px column) to a binary
0 = GOOD / 1 = FAULTY target according to the project's three-tier rules.

Tier 1 – Always GOOD  : embossing, foam_residue, no_fault, water_drop
Tier 2 – Conditionally FAULTY : label is faulty only when area_px > threshold
Tier 3 – Always FAULTY : all remaining defect labels
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Tier definitions ──────────────────────────────────────────────────────────

ALWAYS_GOOD: set[str] = {
    "embossing",
    "foam_residue",
    "no_fault",
    "water_drop",
}

# label → minimum area (px) at which it becomes FAULTY
CONDITIONAL_THRESHOLDS: dict[str, int] = {
    "air_bubble":            500,
    "chip":                  200,
    "contamination_light":   180,
    "glass_imperfection":    100,
    "scuffing":            75_000,
    "scuffing_heavy":        1_200,
}

ALWAYS_FAULTY: set[str] = {
    "break_crack",
    "circlip",
    "contamination_dark",
    "crown_cap",
    "foil_semitransparent",
    "foreign_object_manual",
    "foreign_object_washing",
    "glass_shard",
    "insect",
    "label",
    "liquid",
    "mold",
    "no_base_visible",
    "paint_residue",
    "straw",
    "yeast_residue",
}

# ── Core resolver ─────────────────────────────────────────────────────────────


def resolve_label(label: str, area_px: Optional[float] = None) -> int:
    """
    Return 0 (GOOD) or 1 (FAULTY) for a single annotation row.

    Parameters
    ----------
    label   : raw string label from the dataset (case-insensitive, spaces → _)
    area_px : defect area in pixels; required for conditional labels

    Returns
    -------
    int — 0 = GOOD, 1 = FAULTY
    """
    # Normalise: lowercase, strip, replace spaces/hyphens with underscore
    raw = label.strip().lower().replace(" ", "_").replace("-", "_")

    # Tier 1: always good
    if raw in ALWAYS_GOOD:
        return 0

    # Tier 2: conditional on area
    if raw in CONDITIONAL_THRESHOLDS:
        threshold = CONDITIONAL_THRESHOLDS[raw]
        if area_px is None:
            logger.warning(
                "Label '%s' requires area_px for classification but none provided. "
                "Defaulting to FAULTY (conservative).",
                raw,
            )
            return 1
        return int(area_px > threshold)

    # Tier 3: always faulty
    if raw in ALWAYS_FAULTY:
        return 1

    # Unknown label — conservative default
    logger.warning(
        "Unknown label '%s' (normalised: '%s'). Defaulting to FAULTY.",
        label,
        raw,
    )
    return 1


# ── DataFrame-level helper ────────────────────────────────────────────────────


def resolve_dataframe(
    df: pd.DataFrame,
    label_col: str = "label",
    area_col: str = "area_px",
    target_col: str = "binary_label",
) -> pd.DataFrame:
    """
    Add a binary_label column to an annotations DataFrame.

    Expected columns
    ----------------
    label_col  : str  — raw defect label
    area_col   : float (optional) — defect area in pixels; NaN is fine for
                 labels that don't need it

    Returns the same DataFrame with `target_col` appended (int 0/1).
    """
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in DataFrame.")

    area_series = df[area_col] if area_col in df.columns else pd.Series([None] * len(df))

    df = df.copy()
    df[target_col] = [
        resolve_label(lbl, area)
        for lbl, area in zip(df[label_col], area_series)
    ]

    # Logging summary
    n_good   = (df[target_col] == 0).sum()
    n_faulty = (df[target_col] == 1).sum()
    ratio    = n_faulty / max(n_good, 1)
    logger.info(
        "Label resolution complete — GOOD: %d | FAULTY: %d | imbalance ratio: %.3f",
        n_good,
        n_faulty,
        ratio,
    )
    return df


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    cases = [
        ("no_fault",             None,    0),
        ("water_drop",           5_000,   0),
        ("embossing",            None,    0),
        ("foam_residue",         None,    0),
        ("air_bubble",           300,     0),   # below 500 → good
        ("air_bubble",           600,     1),   # above 500 → faulty
        ("chip",                 150,     0),
        ("chip",                 250,     1),
        ("contamination_light",  100,     0),
        ("contamination_light",  200,     1),
        ("glass_imperfection",   50,      0),
        ("glass_imperfection",   120,     1),
        ("scuffing",             50_000,  0),
        ("scuffing",             80_000,  1),
        ("scuffing_heavy",       1_000,   0),
        ("scuffing_heavy",       1_300,   1),
        ("glass_shard",          0,       1),   # always faulty
        ("insect",               None,    1),
        ("mold",                 None,    1),
        ("Break/Crack",          None,    1),   # normalisation test
    ]

    all_pass = True
    for lbl, area, expected in cases:
        got = resolve_label(lbl, area)
        status = "✓" if got == expected else "✗ FAIL"
        if got != expected:
            all_pass = False
        print(f"  {status}  {lbl!s:30s}  area={str(area):>8}  → {got} (expected {expected})")

    print("\nAll tests passed ✓" if all_pass else "\nSome tests FAILED ✗")
