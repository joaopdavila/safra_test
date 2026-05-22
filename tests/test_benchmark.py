from __future__ import annotations

import math

import pandas as pd

from safra_quant.benchmark import rebase_to


def test_rebase_to_starts_at_base() -> None:
    s = pd.Series([50.0, 60.0, 75.0, 100.0])
    out = rebase_to(s, base=100.0)
    assert math.isclose(out.iloc[0], 100.0)
    assert math.isclose(out.iloc[-1], 200.0)


def test_rebase_preserves_relative_changes() -> None:
    s = pd.Series([10.0, 20.0, 5.0])
    out = rebase_to(s, base=100.0)
    assert math.isclose(out.iloc[1] / out.iloc[0], 2.0)
    assert math.isclose(out.iloc[2] / out.iloc[0], 0.5)


def test_rebase_handles_empty_series() -> None:
    s = pd.Series([], dtype=float)
    out = rebase_to(s)
    assert out.empty
