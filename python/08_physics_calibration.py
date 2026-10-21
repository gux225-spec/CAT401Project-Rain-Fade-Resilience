r"""
Step 8: apply the ideal-physics model (MATLAB simulator) to the measured data.

  A_meas(R) = a * A_ITU(R) + W(R) + e,   W(R) = W_max * (1 - exp(-R / R0))       (wet antenna)
  1. W(R) from the 5 GHz links (control group: ITU rain attenuation at 5 GHz < 0.1 dB)
  2. a from 68-69 GHz after removing W (bin medians, R <= 20 mm/h, fit on Oct-Mar)
  3. cross-band test: the same (a, W) predict 60 GHz and 24 GHz and the held-out Apr-Jun data
  4. physics-based fragility: capacity threshold A_deg per link vs demand A(R) (lognormal, beta from
     residuals) -> P(DS1 | R); compared with the empirical fragility of step 5 on held-out data
  5. drying / recovery time constant tau after rain stops (exponential fit of post-storm tail)
Outputs -> paths.RESULTS / 08_physics_calibration

Run:  python python/08_physics_calibration.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paths import PROCESSED, R2, R5, result_dir
from common import load_link_rain

OUT = result_dir("08_physics_calibration")
SPLIT = pd.Timestamp("2024-04-01")
HEADROOM = 1.5
NOISE = {"5GHz": -88.0, "24GHz": -84.0, "60GHz": -74.0, "68-69GHz": -74.0}
EDGES = np.array([0.5, 1, 2, 3.5, 5, 7.5, 10, 15, 20, 30, 45])


def se(snr_db):
    snr_db = np.asarray(snr_db, float)
    return np.where(snr_db < -10, 0.0, np.minimum(0.6 * np.log2(1 + 10 ** (snr_db / 10)), 4.4))


def ncdf(x):
    z = np.abs(x) / np.sqrt(2); t = 1 / (1 + 0.3275911 * z)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-z * z)
    return 0.5 * (1 + np.sign(x) * y)


LR = load_link_rain(["time_utc", "col", "band", "A_dB", "A_itu_dB", "miss_frac", "rain_mmh"])
LR["col"] = LR.col.astype(str); LR["band"] = LR.band.astype(str)
LR["train"] = LR.time_utc < SPLIT
wet = LR[(LR.rain_mmh >= EDGES[0]) & (LR.miss_frac <= 0.5) & LR.A_dB.notna()].copy()
wet["rb"] = pd.cut(wet.rain_mmh, EDGES, labels=False)
wet = wet[wet.rb.notna()]; wet["rb"] = wet.rb.astype(int)
mid = np.sqrt(EDGES[:-1] * EDGES[1:])


def bin_med(df, col="A_dB"):
    g = df.groupby("rb")
    return pd.DataFrame({"r": mid[g[col].median().index.astype(int)], "A": g[col].median().values, "Aitu": g.A_itu_dB.median().values,
                         "n": g.size().values, "days": g.time_utc.apply(lambda s: s.dt.floor("D").nunique()).values})


# ---------------------------------------------------------------- 1. wet antenna from 5 GHz
b5 = bin_med(wet[(wet.band == "5GHz") & wet.train])
y5 = (b5.A - b5.Aitu).values
best = None
for Wm in np.linspace(0.2, 4, 77):
    for R0 in np.logspace(-1, 1.5, 60):
        e = ((Wm * (1 - np.exp(-b5.r / R0)) - y5) ** 2 * b5.n).sum()
        if best is None or e < best[0]:
            best = (e, Wm, R0)
_, W_MAX, R0 = best
W = lambda r: W_MAX * (1 - np.exp(-np.asarray(r, float) / R0))
print(f"wet antenna (5 GHz control): W_max = {W_MAX:.2f} dB, R0 = {R0:.2f} mm/h")

# ---------------------------------------------------------------- 2. path factor a at 68-69 GHz (R <= 20)
b68 = bin_med(wet[(wet.band == "68-69GHz") & wet.train])
m = b68.r <= 20
x, y = b68.Aitu[m].values, (b68.A[m] - W(b68.r[m])).values
a_hat = (x * y * b68.n[m]).sum() / (x * x * b68.n[m]).sum()
print(f"path/representativeness factor a (68-69 GHz, train) = {a_hat:.2f}")
bt_ = bin_med(wet[(wet.band == "68-69GHz") & ~wet.train]); mt = bt_.r <= 20
a_test = ((bt_.Aitu[mt] * (bt_.A[mt] - W(bt_.r[mt])) * bt_.n[mt]).sum() / (bt_.Aitu[mt] ** 2 * bt_.n[mt]).sum())
print(f"same fit on Apr-Jun only (diagnostic, not used): a = {a_test:.2f}")

# ---------------------------------------------------------------- 3. cross-band & held-out test
rows = []
for band in ["68-69GHz", "60GHz", "24GHz", "5GHz"]:
    for split, df in [("train Oct-Mar", wet[wet.train]), ("test Apr-Jun", wet[~wet.train])]:
        bb = bin_med(df[df.band == band])
        for _, r in bb.iterrows():
            rows.append(dict(band=band, split=split, r=r.r, n=r.n, days=r.days, A_meas=r.A, A_itu=r.Aitu,
                             A_calibrated=a_hat * r.Aitu + float(W(r.r))))
cb = pd.DataFrame(rows)
cb.round(3).to_csv(OUT / "calibration_bins.csv", index=False)
sc = []
for (band, split), g in cb[cb.r <= 20].groupby(["band", "split"]):
    w_ = g.n.values
    sc.append(dict(band=band, split=split,
                   rmse_itu=np.sqrt(np.average((g.A_meas - g.A_itu) ** 2, weights=w_)),
                   rmse_calibrated=np.sqrt(np.average((g.A_meas - g.A_calibrated) ** 2, weights=w_))))
sc = pd.DataFrame(sc); sc.round(3).to_csv(OUT / "calibration_skill.csv", index=False)
print(sc.round(2).to_string())

# ---------------------------------------------------------------- 4. physics-based fragility (DS1) per 68-69 GHz link
meta = pd.read_csv(PROCESSED / "links_meta.csv")
med = pd.read_csv(PROCESSED / "links_rsl_1min.csv.gz", usecols=list(meta.col)).median()
meta["snr_dry"] = meta.col.map(med) - meta.band.map(NOISE)
H = meta[meta.band == "68-69GHz"].copy()
seT = se(H.snr_dry) / HEADROOM
H["A_deg"] = np.where(se(H.snr_dry) / HEADROOM <= 4.4, H.snr_dry - 10 * np.log10(2 ** (seT / 0.6) - 1), 0)
# residual dispersion (train, 68-69 GHz, A_meas > 0.5 dB) around the calibrated prediction
hw = wet[(wet.band == "68-69GHz") & wet.train & (wet.A_dB > 0.5)]
pred = a_hat * hw.A_itu_dB + W(hw.rain_mmh)
BETA = np.std(np.log(hw.A_dB / np.maximum(pred, 0.1)))
print(f"demand dispersion beta = {BETA:.2f}")
RR = np.logspace(-1, np.log10(250), 200)
from_itu = {}
for _, h in H.iterrows():
    sub = LR[(LR.col == h.col) & (LR.rain_mmh > 0.5) & (LR.A_itu_dB > 0)]
    # recover k*L and alpha from the ITU column: A_itu = kL * R^alpha
    lr, la = np.log(sub.rain_mmh), np.log(sub.A_itu_dB)
    al, lkL = np.polyfit(lr, la, 1)
    from_itu[h.col] = (np.exp(lkL), al)
H["kL"] = H.col.map(lambda c: from_itu[c][0]); H["alpha"] = H.col.map(lambda c: from_itu[c][1])


def p_ds1_phys(R, beta=BETA, a=a_hat):
    out = []
    for _, h in H.iterrows():
        Ad = a * h.kL * np.asarray(R) ** h.alpha + W(R)
        out.append(ncdf(np.log(np.maximum(Ad, 1e-6) / max(h.A_deg, 1e-3)) / beta))
    return np.mean(out, axis=0)


emp = pd.read_csv(R5 / "fragility_parameters.csv")
e68 = emp[(emp.band == "68-69GHz") & (emp.DS == "DS1")].iloc[0]
p0 = e68.p0                                                      # dry-weather floor (non-rain causes), train data
frag_phys = p0 + (1 - p0) * p_ds1_phys(RR)
frag_emp = e68.p0 + (1 - e68.p0) * ncdf(np.log(RR / e68.theta_mmh) / e68.beta)
rel = pd.read_csv(R5 / "validation_reliability_test.csv")
rel = rel[(rel.band == "68-69GHz") & (rel.DS == "DS1") & (rel.r_med > 0)]
# Brier on held-out wet windows: physics vs empirical
te = LR[(LR.band == "68-69GHz") & ~LR.train & (LR.rain_mmh > 0)].merge(meta[["col", "snr_dry"]], on="col")
g = np.where(te.miss_frac > 0.5, 0, se(te.snr_dry - te.A_dB.fillna(0).clip(lower=0)) / se(te.snr_dry))
yb = (HEADROOM * g < 1).astype(float)
pp_phys = np.interp(te.rain_mmh, RR, frag_phys)
pp_emp = np.interp(te.rain_mmh, RR, frag_emp)
bs = dict(physics=np.mean((pp_phys - yb) ** 2), empirical=np.mean((pp_emp - yb) ** 2), climatology=np.mean((yb.mean() - yb) ** 2))
print("Brier held-out (68-69 GHz DS1):", {k: round(v, 4) for k, v in bs.items()})
theta_phys = np.interp(0.5, (frag_phys - p0) / (1 - p0), RR)

# ---------------------------------------------------------------- 5. drying time constant
ev = pd.read_csv(R2 / "rain_events.csv", parse_dates=["start", "end"])
ev = ev[~ev.likely_snow_or_mixed]
taus = []
for band in ["68-69GHz", "5GHz"]:
    B = LR[LR.band == band].set_index("time_utc").sort_index()
    for _, r in ev.iterrows():
        t0, t1 = r.end, r.end + pd.Timedelta("3h")
        for c, gg in B.loc[t0:t1].groupby("col"):
            if len(gg) < 24 or (gg.rain_mmh > 0).any() or (gg.miss_frac > 0.5).any():
                continue
            a_ = gg.A_dB.values.astype(float); tt = (gg.index - t0).total_seconds().values / 60
            if np.nanmax(a_[:3]) < 1.0 or np.isnan(a_).mean() > 0.2:
                continue
            ok = ~np.isnan(a_)
            # A(t) = c + A0 exp(-t/tau): grid over tau, linear least squares for (c, A0)
            bestf = None
            for tau in np.logspace(np.log10(5), np.log10(600), 80):
                X_ = np.column_stack([np.ones(ok.sum()), np.exp(-tt[ok] / tau)])
                coef, res, *_ = np.linalg.lstsq(X_, a_[ok], rcond=None)
                sse = ((X_ @ coef - a_[ok]) ** 2).sum()
                if coef[1] > 0.5 and (bestf is None or sse < bestf[0]):
                    bestf = (sse, tau, coef[0], coef[1])
            if bestf is None:
                continue
            ex = a_ - np.nanmedian(a_[-6:])
            t50 = tt[np.argmax(ex <= 0.5 * np.nanmax(ex[:3]))] if (ex <= 0.5 * np.nanmax(ex[:3])).any() else np.nan
            taus.append(dict(band=band, col=c, storm=r.start, A0=bestf[3], offset=bestf[2], tau_min=bestf[1], t50_min=t50))
taus = pd.DataFrame(taus)
taus.round(2).to_csv(OUT / "drying_time_constants.csv", index=False)
tq = taus.groupby("band").tau_min.describe(percentiles=[.25, .5, .75]).round(1)
print(tq); print(taus.groupby("band").t50_min.describe(percentiles=[.25, .5, .75]).round(1))
t50 = taus.groupby("band").t50_min.median()
summary = pd.Series(dict(W_max_dB=W_MAX, R0_mmh=R0, a_path=a_hat, a_path_AprJun_diagnostic=a_test, beta_demand=BETA,
                         theta_phys_DS1=theta_phys, theta_emp_DS1=e68.theta_mmh,
                         t50_68_median_min=t50.get("68-69GHz", np.nan), t50_5_median_min=t50.get("5GHz", np.nan),
                         tau_dry_68_median_min=tq.loc["68-69GHz", "50%"] if "68-69GHz" in tq.index else np.nan,
                         tau_dry_5_median_min=tq.loc["5GHz", "50%"] if "5GHz" in tq.index else np.nan,
                         n_tau_68=int((taus.band == "68-69GHz").sum()), brier_physics=bs["physics"],
                         brier_empirical=bs["empirical"], brier_climatology=bs["climatology"]))
summary.round(4).to_csv(OUT / "calibrated_parameters.csv")
print(summary.round(3))

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
colors = {"68-69GHz": "tab:red", "60GHz": "tab:purple", "24GHz": "tab:green", "5GHz": "tab:blue"}
for band, c in colors.items():
    for split, mk, fill in [("train Oct-Mar", "o", c), ("test Apr-Jun", "s", "white")]:
        q = cb[(cb.band == band) & (cb.split == split) & (cb.n >= 30)]
        show = band in ("68-69GHz", "5GHz") or split.startswith("train")
        ax[0].plot(q.r, q.A_meas, mk, color=c, mfc=fill, ms=5, label=f"{band} measured ({split})" if show else None)
    q = cb[(cb.band == band) & (cb.split == "train Oct-Mar")]
    ax[0].plot(q.r, q.A_calibrated, "-", color=c, lw=1.5)
    ax[0].plot(q.r, q.A_itu, ":", color=c, lw=1)
ax[0].plot([], [], "k-", label=f"calibrated: {a_hat:.2f} x ITU + W(R)"); ax[0].plot([], [], "k:", label="ITU-R P.838 only")
ax[0].set(xscale="log", xlabel="15-min rain rate at link (mm/h)", ylabel="median extra loss (dB)",
          title=f"(a) Calibrated physics: a={a_hat:.2f}, W_max={W_MAX:.1f} dB, R0={R0:.1f} mm/h")
ax[0].legend(fontsize=6.3, loc="upper left"); ax[0].grid(alpha=.3, which="both")
ax[1].plot(RR, frag_phys, "-", color="tab:red", lw=2, label=f"physics-based (capacity vs demand), median R = {theta_phys:.1f} mm/h")
ax[1].plot(RR, frag_emp, "--", color="tab:orange", lw=1.5, label=f"empirical lognormal (step 5), median R = {e68.theta_mmh:.1f} mm/h")
ax[1].errorbar(rel.r_med, rel.obs, yerr=[rel.obs - rel.obs_lo, rel.obs_hi - rel.obs], fmt="ko", ms=4, capsize=2,
               label="observed, held-out Apr-Jun")
ax[1].set(xscale="log", xlabel="15-min rain rate (mm/h)", ylabel="P(DS1: high band cannot carry demand)",
          title=f"(b) Fragility, 68-69 GHz: held-out Brier physics {bs['physics']:.3f} / empirical {bs['empirical']:.3f}")
ax[1].legend(fontsize=6.8, loc="upper left"); ax[1].grid(alpha=.3, which="both")
for band, c in [("68-69GHz", "tab:red"), ("5GHz", "tab:blue")]:
    v = taus[taus.band == band].t50_min.dropna()
    ax[2].hist(v, bins=np.arange(0, 125, 5), alpha=.6, color=c,
               label=f"{band}: median {v.median():.0f} min, IQR {v.quantile(.25):.0f}-{v.quantile(.75):.0f} (n={len(v)})")
ax[2].set(xlabel="time to recover 50 % of the excess loss after rain stops (min)", ylabel="link-storm count",
          title="(c) Measured post-storm recovery (wet-antenna drying)")
ax[2].legend(fontsize=7); ax[2].grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "fig13_physics_calibration.png", dpi=150)
print("done ->", OUT)
