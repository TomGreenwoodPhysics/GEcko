"""
Tests for gecko.stats.cointegration.

Two groups:
  * NUMPY-ONLY tests (TestHedgeAndSpread): exercise ols_hedge_ratio,
    build_spread, spread_half_life. No statsmodels needed -- these are the
    parts whose math we control directly.
  * STATSMODELS tests (TestCointegrationTests): exercise adf / engle_granger /
    johansen against synthetic series with a known ground truth (a
    cointegrated pair vs two independent random walks). These require
    statsmodels installed.

Run everything:        pytest tests/test_cointegration.py -v
Run only numpy parts:  pytest tests/test_cointegration.py -v -k HedgeAndSpread
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the numpy-only helpers directly so this part can be collected even if
# statsmodels is absent (the module raises on import without it, so guard).
statsmodels_available = True
try:
    from gecko.stats import cointegration as co
except ImportError:
    statsmodels_available = False
    co = None


# ---------------------------------------------------------------------------
# Numpy-only: hedge ratio, spread, half-life
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not statsmodels_available,
                    reason="module import needs statsmodels present")
class TestHedgeAndSpread:

    def test_ols_recovers_known_beta(self):
        rng = np.random.default_rng(0)
        x = pd.Series(rng.normal(size=2000).cumsum())
        y = 3.0 + 2.5 * x + rng.normal(scale=0.01, size=2000)  # beta=2.5, alpha=3
        out = co.ols_hedge_ratio(y, x)
        assert abs(out["beta"] - 2.5) < 0.01
        assert abs(out["alpha"] - 3.0) < 0.05

    def test_build_spread_is_residual(self):
        idx = pd.date_range("2020-01-01", periods=100)
        log_raw = pd.Series(np.linspace(1, 2, 100), index=idx)
        log_proc = 0.5 + 1.0 * log_raw  # perfectly explained
        spread = co.build_spread(log_raw, log_proc, beta=1.0, alpha=0.5)
        assert np.allclose(spread.values, 0.0, atol=1e-9)

    def test_half_life_of_known_ar1(self):
        # AR(1) with phi=0.5 has half-life -ln2/ln(0.5) = exactly 1 step.
        rng = np.random.default_rng(1)
        n = 20000
        s = np.zeros(n)
        for t in range(1, n):
            s[t] = 0.5 * s[t-1] + rng.normal(scale=0.1)
        hl = co.spread_half_life(pd.Series(s))
        assert abs(hl - 1.0) < 0.1

    def test_half_life_large_or_nan_for_random_walk(self):
        # A pure random walk has phi just under 1 (finite-sample bias), so the
        # half-life comes out very LARGE rather than NaN. The contract is only
        # that it must not look like a fast-reverting spread -- this is exactly
        # why mean-reversion is decided by the cointegration test, not here.
        rng = np.random.default_rng(2)
        s = pd.Series(rng.normal(size=5000).cumsum())
        hl = co.spread_half_life(s)
        assert np.isnan(hl) or hl > 100


# ---------------------------------------------------------------------------
# Statsmodels: the actual cointegration tests, against known ground truth
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not statsmodels_available,
                    reason="requires statsmodels")
class TestCointegrationTests:

    @staticmethod
    def _cointegrated_pair(n=3000, beta=1.0, seed=0):
        """raw = random walk (I(1)); proc = beta*raw + stationary noise.
        By construction these ARE cointegrated with hedge ratio beta."""
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2015-01-01", periods=n)
        log_raw = pd.Series(rng.normal(scale=0.02, size=n).cumsum() + 4.0, index=idx)
        noise = pd.Series(np.zeros(n), index=idx)
        nv = noise.values
        for t in range(1, n):  # stationary AR(1) noise
            nv[t] = 0.9 * nv[t-1] + rng.normal(scale=0.01)
        log_proc = beta * log_raw + 0.3 + noise
        return log_raw, log_proc

    @staticmethod
    def _independent_walks(n=3000, seed=0):
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2015-01-01", periods=n)
        a = pd.Series(rng.normal(scale=0.02, size=n).cumsum() + 4.0, index=idx)
        b = pd.Series(rng.normal(scale=0.02, size=n).cumsum() + 5.0, index=idx)
        return a, b

    def test_integration_order_of_random_walk_is_I1(self):
        log_raw, _ = self._cointegrated_pair()
        order = co.integration_order(log_raw)
        assert order["verdict"] == "I(1)"

    def test_engle_granger_detects_cointegration(self):
        log_raw, log_proc = self._cointegrated_pair(beta=1.0)
        eg = co.engle_granger(log_raw, log_proc)
        assert eg["proc_on_raw"]["cointegrated_at_5pct"]
        assert abs(eg["recommended_beta"] - 1.0) < 0.1

    def test_engle_granger_rejects_independent_walks(self):
        a, b = self._independent_walks()
        eg = co.engle_granger(a, b)
        # two independent random walks should NOT look cointegrated
        assert not eg["proc_on_raw"]["cointegrated_at_5pct"]

    def test_johansen_detects_cointegration(self):
        log_raw, log_proc = self._cointegrated_pair(beta=1.0)
        joh = co.johansen(log_raw, log_proc)
        assert joh["trace_rejects_r0_at_5pct"]
        assert abs(joh["beta_johansen"] - 1.0) < 0.25

    def test_run_pair_cointegration_end_to_end(self):
        log_raw, log_proc = self._cointegrated_pair(beta=1.0)
        panel = pd.DataFrame({"raw_price": np.exp(log_raw),
                              "proc_price": np.exp(log_proc)})
        rep = co.run_pair_cointegration(panel)
        assert rep["engle_granger"]["proc_on_raw"]["cointegrated_at_5pct"]
        assert np.isfinite(rep["spread_half_life_days"])
        assert isinstance(rep["spread"], pd.Series)


if __name__ == "__main__":
    # Allow running the numpy-only group without pytest, for a quick check.
    if not statsmodels_available:
        print("statsmodels not importable; cannot run.")
        sys.exit(0)
    t = TestHedgeAndSpread()
    t.test_ols_recovers_known_beta()
    t.test_build_spread_is_residual()
    t.test_half_life_of_known_ar1()
    t.test_half_life_nan_for_random_walk()
    print("numpy-only tests passed")