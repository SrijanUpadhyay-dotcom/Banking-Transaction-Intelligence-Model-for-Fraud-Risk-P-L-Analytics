# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Fraud-loss forecasting: 30/60/90-day losses with prediction intervals, by
jurisdiction and by channel.

**Model (per slice, e.g. one country).**
- *Frequency.* Daily confirmed-fraud counts follow an over-dispersed Poisson
  model. Log rate = intercept + day-of-week + linear trend. The trend is
  shrunk towards zero, so a noisy year does not extrapolate into a runaway
  forecast. It is fitted on the trailing year by penalised IRLS.
  Over-dispersion φ (Pearson χ²) is simulated as a gamma-Poisson mixture.
  Day-of-week and trend terms are dropped for sparse slices: below 300 or 100
  frauds in the fit window.
- *Severity.* Realised loss per fraud, after recovery, in USD. Fraud losses
  are extremely heavy-tailed: on the synthetic book the largest 1% of frauds
  carry 45% of the loss, and half are fully recovered. Each slice draws from
  its own losses over the trailing year (recent severity: the backtest showed
  all-history pools miss a rise in loss size), shrunk towards the portfolio pool
  when the slice has few frauds. Every draw above the portfolio's 90th
  percentile of positive losses is replaced by a draw from a generalised
  Pareto tail fitted to the exceedances. A future loss can then exceed the
  worst one seen so far. Draws are capped at 3× the largest observed loss,
  standing in for the bank's transaction limits. Each path also scales its
  losses by a bootstrap ratio of the slice's mean loss, so the uncertainty in
  the average loss is carried into long horizons instead of averaging away.
- *Uncertainty.* Every Monte Carlo path also draws the model coefficients from
  their estimated sampling distribution. The intervals therefore cover
  estimation error as well as day-to-day randomness.
- *Coherence.* The total is the path-by-path sum of the country forecasts, so
  the countries add up to the total. Channels are a second, independent
  partition.

**Label maturity (IBNR).** Fraud is confirmed days or weeks after it happens.
The last few weeks of history are therefore incomplete, and a naive model
would forecast a fall that is not real. Where confirmation times exist,
`completion_factors` estimates the confirmation-delay distribution from
mature history. Known counts are then divided by the share expected to be
confirmed by the forecast date, and days less than half complete are left out
of the fit. This is the chain-ladder idea from insurance reserving. Without
confirmation times, history is treated as complete, and the report says so.

**Backtest.** `backtest` re-runs the forecast from rolling monthly origins.
Each run uses only what was known at its origin. It reports interval coverage
and the median error against a naive baseline: trailing-90-day mean daily
loss × horizon.

    python -m bti.planning.forecast            # forecast from the latest date, with backtest
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from bti.planning.history import known_as_of

HORIZONS = (30, 60, 90)
OUT = Path("outputs/planning/loss_forecast.json")
QUANTILES = {"p05": 5, "p10": 10, "p50": 50, "p90": 90, "p95": 95}


# ── label maturity ───────────────────────────────────────────────────────────
def completion_factors(history: pd.DataFrame, as_of: pd.Timestamp, max_delay_days: int = 180,
                       min_frauds: int = 50) -> Optional[np.ndarray]:
    """F[a] = share of fraud confirmed within `a` days, from frauds old enough to be fully reported."""
    mature = history[(history["fraud"] == 1) & history["confirmed_at"].notna()
                     & (history["date"] <= as_of - pd.Timedelta(days=max_delay_days))]
    if len(mature) < min_frauds:
        return None
    delay = (mature["confirmed_at"].dt.normalize() - mature["date"]).dt.days.clip(lower=0, upper=max_delay_days)
    counts = np.bincount(delay.to_numpy(int), minlength=max_delay_days + 1)
    return np.cumsum(counts) / counts.sum()


def daily(history: pd.DataFrame, as_of: pd.Timestamp, start: pd.Timestamp, by: Optional[str] = None) -> pd.DataFrame:
    """Daily known fraud count, loss and volume on [start, as_of], one block of columns per slice value."""
    h = known_as_of(history, as_of)
    h = h[h["date"] >= start]
    days = pd.date_range(start, as_of, freq="D")
    key = h[by].fillna("(unknown)") if by else pd.Series("total", index=h.index)
    g = h.assign(_k=key).groupby(["_k", "date"])
    frame = pd.DataFrame({"count": g["fraud"].sum(), "loss": g["loss_usd"].sum(), "volume": g.size()})
    out = {}
    for k in sorted(key.unique()):
        part = frame.loc[k].reindex(days, fill_value=0) if k in frame.index.get_level_values(0) else \
            pd.DataFrame(0, index=days, columns=["count", "loss", "volume"])
        out[k] = part
    return pd.concat(out, axis=1)


# ── frequency model ──────────────────────────────────────────────────────────
class CountModel:
    """Over-dispersed Poisson regression on day-of-week and a shrunk linear trend (penalised IRLS)."""

    def __init__(self, dow: bool = True, trend: bool = True, trend_ridge: float = 4.0):
        self.dow, self.trend, self.trend_ridge = dow, trend, trend_ridge

    def _design(self, dates: pd.DatetimeIndex) -> np.ndarray:
        cols = [np.ones(len(dates))]
        if self.dow:
            wd = dates.dayofweek.to_numpy()
            cols += [(wd == d).astype(float) for d in range(1, 7)]
        if self.trend:
            cols.append(((dates - self.origin).days.to_numpy() / 365.25))
        return np.column_stack(cols)

    def fit(self, dates: pd.DatetimeIndex, y: np.ndarray, weight: Optional[np.ndarray] = None) -> "CountModel":
        self.origin = dates[-1]
        X = self._design(dates)
        w0 = np.ones(len(y)) if weight is None else weight
        penalty = np.zeros(X.shape[1])
        penalty[1:] = 1e-3
        if self.trend:
            penalty[-1] = self.trend_ridge
        beta = np.zeros(X.shape[1])
        beta[0] = np.log(max(np.average(y, weights=w0), 1e-3))
        for _ in range(50):
            mu = np.exp(X @ beta)
            W = w0 * mu
            H = X.T @ (X * W[:, None]) + np.diag(penalty)
            grad = X.T @ (w0 * (y - mu)) - penalty * beta
            step = np.linalg.solve(H, grad)
            beta += step
            if np.max(np.abs(step)) < 1e-8:
                break
        mu = np.exp(X @ beta)
        dof = max(len(y) - X.shape[1], 1)
        self.phi = float(max(np.sum(w0 * (y - mu) ** 2 / np.maximum(mu, 1e-9)) / dof, 1.0))
        self.beta = beta
        self.cov = np.linalg.inv(X.T @ (X * (w0 * mu)[:, None]) + np.diag(penalty)) * self.phi
        self.mean_rate = float(np.average(y, weights=w0))
        return self

    def simulate(self, dates: pd.DatetimeIndex, n_paths: int, rng: np.random.Generator) -> np.ndarray:
        X = self._design(dates)
        betas = rng.multivariate_normal(self.beta, self.cov, size=n_paths, method="cholesky")
        mu = np.exp(np.clip(betas @ X.T, -20, 10))
        if self.phi > 1.0001:
            lam = rng.gamma(shape=mu / (self.phi - 1), scale=self.phi - 1)
        else:
            lam = mu
        return rng.poisson(lam)

    def describe(self) -> Dict:
        return {"mean_frauds_per_day": round(self.mean_rate, 3), "dispersion": round(self.phi, 3),
                "day_of_week": self.dow,
                "trend_per_year_pct": round(float(np.expm1(self.beta[-1])) * 100, 1) if self.trend else None}


# ── severity ─────────────────────────────────────────────────────────────────
class Severity:
    """Empirical losses with a generalised-Pareto tail above the 90th percentile of positive losses."""

    def __init__(self, losses: np.ndarray, tail_quantile: float = 0.90, min_exceedances: int = 30,
                 cap_multiple: float = 3.0):
        from scipy.stats import genpareto
        positive = losses[losses > 0]
        self.threshold, self.tail = None, None
        self.cap = float(losses.max() * cap_multiple) if len(losses) else 0.0
        if len(positive) >= min_exceedances / (1 - tail_quantile):
            u = float(np.quantile(positive, tail_quantile))
            exceed = positive[positive > u] - u
            xi, _, sigma = genpareto.fit(exceed, floc=0)
            self.threshold, self.tail = u, (float(xi), float(sigma))

    def extend(self, draws: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.threshold is None:
            return draws
        from scipy.stats import genpareto
        mask = draws > self.threshold
        n = int(mask.sum())
        if n:
            xi, sigma = self.tail
            draws = draws.copy()
            draws[mask] = np.minimum(self.threshold + genpareto.rvs(xi, scale=sigma, size=n, random_state=rng),
                                     self.cap)
        return draws

    def describe(self) -> Dict:
        if self.threshold is None:
            return {"tail": "empirical only (too few large losses to fit a tail)"}
        return {"tail": "generalised Pareto", "threshold_usd": round(self.threshold, 0),
                "shape_xi": round(self.tail[0], 3), "scale_usd": round(self.tail[1], 0), "cap_usd": round(self.cap, 0)}


# ── forecast ─────────────────────────────────────────────────────────────────
def _stats(a: np.ndarray, digits: int = 0) -> Dict:
    out = {"mean": round(float(a.mean()), digits)}
    out.update({k: round(float(np.percentile(a, q)), digits) for k, q in QUANTILES.items()})
    return out


def _severity_paths(counts: np.ndarray, pool: np.ndarray, global_pool: np.ndarray, shrink: float,
                    horizons: Sequence[int], rng: np.random.Generator, severity: Optional[Severity] = None) -> np.ndarray:
    """Cumulative loss at each horizon for every path: sums of `counts` severity draws."""
    n_paths = counts.shape[0]
    cum_counts = np.cumsum(counts, axis=1)[:, [h - 1 for h in horizons]]       # frauds by each horizon
    total = int(cum_counts[:, -1].sum())
    if total == 0:
        return np.zeros((n_paths, len(horizons)))
    own = rng.random(total) < shrink if len(pool) else np.zeros(total, bool)
    draws = np.where(own, pool[rng.integers(0, max(len(pool), 1), total)] if len(pool) else 0.0,
                     global_pool[rng.integers(0, len(global_pool), total)])
    if severity is not None:
        draws = severity.extend(draws, rng)
    # severity-level uncertainty: each path's losses are scaled by a bootstrap ratio of the pool's mean, so the
    # uncertainty in the average loss does not average away over long horizons
    base = pool if len(pool) >= 30 else global_pool
    if len(base) > 1 and base.mean() > 0:
        n = min(len(base), 2000)
        level = base[rng.integers(0, len(base), (n_paths, n))].mean(axis=1) / base.mean()
        draws = draws * np.repeat(level, cum_counts[:, -1])
    cs = np.concatenate([[0.0], np.cumsum(draws)])
    offsets = np.concatenate([[0], np.cumsum(cum_counts[:, -1])[:-1]])
    return cs[offsets[:, None] + cum_counts] - cs[offsets][:, None]


def _fit_slice(series: pd.DataFrame, factors: Optional[np.ndarray], as_of: pd.Timestamp):
    y, loss = series["count"].to_numpy(float), series["loss"].to_numpy(float)
    dates = series.index
    if factors is not None:
        age = (as_of - dates).days.to_numpy()
        f = factors[np.minimum(age, len(factors) - 1)]
        keep = f >= 0.5
        y = np.where(keep, y / np.maximum(f, 1e-9), 0)
        loss = np.where(keep, loss / np.maximum(f, 1e-9), 0)
        dates, y, loss = dates[keep], y[keep], loss[keep]
    n = y.sum()
    return CountModel(dow=n >= 300, trend=n >= 100).fit(dates, y), dates


def forecast(history: pd.DataFrame, as_of: Optional[pd.Timestamp] = None, horizons: Sequence[int] = HORIZONS,
             fit_days: int = 365, n_paths: int = 4000, seed: int = 0,
             partitions: Sequence[str] = ("country", "channel"), severity_days: Optional[int] = 365) -> Dict:
    as_of = pd.Timestamp(as_of or history["date"].max()).normalize()
    start = as_of - pd.Timedelta(days=fit_days - 1)
    rng = np.random.default_rng(seed)
    factors = completion_factors(history, as_of)
    known = known_as_of(history, as_of)
    window = known[known["fraud"] == 1]                          # severity pool: trailing year by default
    if severity_days:
        window = window[window["date"] > as_of - pd.Timedelta(days=severity_days)]
    global_pool = window["loss_usd"].to_numpy(float)
    if not ((known["date"] >= start) & (known["fraud"] == 1)).any():
        raise ValueError("no confirmed fraud in the fit window")
    severity = Severity(global_pool)
    future = pd.date_range(as_of + pd.Timedelta(days=1), periods=max(horizons), freq="D")
    result = {"as_of": as_of.date().isoformat(), "fit_window": [start.date().isoformat(), as_of.date().isoformat()],
              "horizons_days": list(horizons), "currency": "USD", "paths": n_paths,
              "loss_definition": "realised fraud loss after recovery (fraud_loss), converted at reference FX rates",
              "label_maturity": ({"method": "completion factors from confirmation delays",
                                  "share_confirmed_within_days": {d: round(float(factors[d]), 3)
                                                                  for d in (7, 14, 30, 60, 90) if d < len(factors)}}
                                 if factors is not None else
                                 {"method": "none", "note": "no confirmation times: recent history treated as "
                                                            "complete. In a bank, feed label confirmation dates so "
                                                            "the latest weeks are grossed up."})}
    for part in partitions:
        series = daily(history, as_of, start, by=part)
        paths_by_slice, slices, models = {}, {}, {}
        count_paths_total = None
        for value in series.columns.get_level_values(0).unique():
            s = series[value]
            model, _ = _fit_slice(s, factors, as_of)
            counts = model.simulate(future, n_paths, rng)
            pool = window.loc[window[part].fillna("(unknown)") == value, "loss_usd"].to_numpy(float)
            shrink = len(pool) / (len(pool) + 30.0)
            loss_paths = _severity_paths(counts, pool, global_pool, shrink, horizons, rng, severity)
            cnt = np.cumsum(counts, axis=1)[:, [h - 1 for h in horizons]]
            paths_by_slice[value] = loss_paths
            count_paths_total = cnt if count_paths_total is None else count_paths_total + cnt
            models[value] = model.describe()
            slices[value] = {f"{h}d": {"loss_usd": _stats(loss_paths[:, i]), "frauds": _stats(cnt[:, i], 1)}
                             for i, h in enumerate(horizons)}
        total = sum(paths_by_slice.values())
        result[f"by_{part}"] = slices
        result[f"models_by_{part}"] = models
        result.setdefault("total", {f"{h}d": {"loss_usd": _stats(total[:, i]),
                                              "frauds": _stats(count_paths_total[:, i], 1)}
                                    for i, h in enumerate(horizons)})
        result.setdefault("total_from", part)
    result["severity"] = {"frauds_in_pool": int(len(global_pool)), **severity.describe()}
    hist_loss = known[known["date"] >= start]["loss_usd"].sum()
    result["trailing_year_loss_usd"] = round(float(hist_loss), 0)
    return result


# ── backtest ─────────────────────────────────────────────────────────────────
def backtest(history: pd.DataFrame, origins: Optional[List[pd.Timestamp]] = None, horizons: Sequence[int] = HORIZONS,
             fit_days: int = 365, n_paths: int = 2000, partition: str = "country",
             severity_days: Optional[int] = 365) -> Dict:
    """Rolling-origin evaluation: each origin forecasts from what was known then and is scored on what happened."""
    last = history["date"].max()
    if origins is None:
        first = history["date"].min() + pd.Timedelta(days=fit_days)
        origins = list(pd.date_range(first, last - pd.Timedelta(days=max(horizons)), freq="30D"))
    rows = []
    for k, origin in enumerate(origins):
        f = forecast(history, origin, horizons, fit_days, n_paths, seed=k, partitions=(partition,),
                     severity_days=severity_days)
        known = known_as_of(history, origin)
        trailing = known[known["date"] > origin - pd.Timedelta(days=90)]["loss_usd"].sum() / 90.0
        for scope, block in [("total", f["total"])] + [(f"{partition}:{v}", s) for v, s in f[f"by_{partition}"].items()]:
            mask = history["date"] > origin
            if scope != "total":
                mask &= history[partition].fillna("(unknown)") == scope.split(":", 1)[1]
            for h in horizons:
                actual = float(history.loc[mask & (history["date"] <= origin + pd.Timedelta(days=h)), "loss_usd"].sum())
                b = block[f"{h}d"]["loss_usd"]
                row = {"origin": origin.date().isoformat(), "scope": scope, "horizon": h, "actual": actual,
                       "p50": b["p50"], "in80": b["p10"] <= actual <= b["p90"], "in90": b["p05"] <= actual <= b["p95"],
                       "below80": actual < b["p10"], "above80": actual > b["p90"]}
                if scope == "total":
                    row["baseline"] = trailing * h
                rows.append(row)
    r = pd.DataFrame(rows)
    tot = r[r["scope"] == "total"]

    def ape(pred, actual):
        return float(np.median(np.abs(pred - actual) / np.maximum(actual, 1.0)))
    summary = {
        "origins": len(origins), "first_origin": origins[0].date().isoformat(), "last_origin": origins[-1].date().isoformat(),
        "total": {f"{h}d": {"coverage_80pct_interval": round(float(tot[tot.horizon == h]["in80"].mean()), 3),
                            "coverage_90pct_interval": round(float(tot[tot.horizon == h]["in90"].mean()), 3),
                            "misses_below_above_80pct": [int(tot[tot.horizon == h]["below80"].sum()),
                                                         int(tot[tot.horizon == h]["above80"].sum())],
                            "median_abs_pct_error_forecast": round(ape(tot[tot.horizon == h]["p50"],
                                                                       tot[tot.horizon == h]["actual"]), 3),
                            "median_abs_pct_error_naive": round(ape(tot[tot.horizon == h]["baseline"],
                                                                    tot[tot.horizon == h]["actual"]), 3)}
                  for h in horizons},
        f"by_{partition}_pooled": {f"{h}d": {
            "coverage_80pct_interval": round(float(r[(r.scope != "total") & (r.horizon == h)]["in80"].mean()), 3),
            "coverage_90pct_interval": round(float(r[(r.scope != "total") & (r.horizon == h)]["in90"].mean()), 3),
            "misses_below_above_80pct": [int(r[(r.scope != "total") & (r.horizon == h)]["below80"].sum()),
                                         int(r[(r.scope != "total") & (r.horizon == h)]["above80"].sum())],
            "n": int(((r.scope != "total") & (r.horizon == h)).sum())} for h in horizons},
    }
    return summary


def run(path: Optional[str] = None, with_backtest: bool = True) -> Dict:
    from bti.planning.history import load
    history = load(path)
    report = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "forecast": forecast(history)}
    if with_backtest:
        report["backtest"] = backtest(history)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str))
    return report


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Fraud-loss forecast (30/60/90 days) with backtest")
    parser.add_argument("--data", default=None)
    parser.add_argument("--no-backtest", action="store_true")
    args = parser.parse_args()
    r = run(args.data, not args.no_backtest)
    f = r["forecast"]
    print(f"as of {f['as_of']}  (trailing-year loss ${f['trailing_year_loss_usd']:,.0f})")
    for h, b in f["total"].items():
        l = b["loss_usd"]
        print(f"  {h:>4}: ${l['p50']:>12,.0f}  80% [{l['p10']:,.0f} – {l['p90']:,.0f}]  "
              f"frauds {b['frauds']['p50']:.0f}")
    if "backtest" in r:
        print(json.dumps(r["backtest"], indent=1))


if __name__ == "__main__":
    main()
