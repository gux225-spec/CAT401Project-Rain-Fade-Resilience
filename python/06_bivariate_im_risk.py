r"""
Step 6: two-parameter intensity measure (peak 15-min rain rate P + storm depth D).

Why: step 5 described each storm by its peak only, so long, moderate storms (e.g. 2023-12-18,
59 mm over 19 h) were under-predicted; and step 5's hazard was flat below u = 10 mm/h, so storms
with 1-10 mm/h peaks were never simulated. Both are fixed here. Everything is at SITE (point)
scale, so hazard, loss model and NOAA Atlas 14 use the same IM.

  site IM       P = max 15-min rain rate at the link, D = max 24-h accumulation overlapping the storm
                (fixed duration, same definition as Atlas 14 24-h depth)
                (3 nearest QC'd PWS, from step 2)
  loss model    ln(1+EOM) = b0 + b1 ln P [+ b2 ln D] + e   per site and storm
                M1 peak only vs M2 peak + depth, validated leave-one-STORM-out
  hazard        marginals per station-year from ASOS + QC'd PWS station-storm values
                  P: empirical < 10 mm/h, Poisson-GPD 10-64, NOAA Atlas 14 15-min >= 1-yr
                  D: empirical < 40 mm, bridge to Atlas 14 24-h 1-yr, Atlas 14 24-h beyond
                dependence: Gaussian copula, rho from station-storm Spearman correlation
                duration: ln H = h0 + h1 ln P + h2 ln D (observed), used to cap EOM
  risk          stochastic storm set -> EOM per site -> annual exceedance frequency, EAL
  validation    LOEO loss skill; P and D marginals vs Atlas 14; joint frequency check;
                back-test of the observed-only model against the 8-month record

Outputs -> paths.RESULTS / 06_bivariate_im

Run:  python python/06_bivariate_im_risk.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paths import ASOS_CSV, ATLAS_INTENSITY, ATLAS_DEPTH, PROCESSED, R2, result_dir
from common import load_link_rain

OUT = result_dir("06_bivariate_im")
rng = np.random.default_rng(401)

HEADROOM = 1.5
S_LIST = [0.0, 0.25, 0.5, 1.0]
NOISE = {"5GHz": -88.0, "68-69GHz": -74.0}
T_OBS = (pd.Timestamp("2024-07-01") - pd.Timestamp("2023-10-29")).days / 365.25
N_SIM, N_BOOT, U, P_MIN = 200_000, 100, 10.0, 1.0
MODELS = ["M1 peak only", "M2 peak + depth"]


def se(snr_db):
    snr_db = np.asarray(snr_db, float)
    return np.where(snr_db < -10, 0.0, np.minimum(0.6 * np.log2(1 + 10 ** (snr_db / 10)), 4.4))


def ncdf(x):
    z = np.abs(x) / np.sqrt(2); t = 1 / (1 + 0.3275911 * z)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-z * z)
    return 0.5 * (1 + np.sign(x) * y)


def ols(X, y):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b
    return b, np.sqrt(r @ r / max(len(y) - X.shape[1], 1))


def design(P, D, model):
    cols = [np.ones(len(P)), np.log(P)] + ([np.log(D)] if model == MODELS[1] else [])
    return np.column_stack(cols)


# ============================================================ storms
ev = pd.read_csv(R2 / "rain_events.csv", parse_dates=["start", "end"])
ev = ev[~ev.likely_snow_or_mixed].reset_index(drop=True)
ev["w0"], ev["w1"] = ev.start - pd.Timedelta("30min"), ev.end + pd.Timedelta("30min")

# ============================================================ site-level loss per storm (68-69 GHz -> 5 GHz)
meta = pd.read_csv(PROCESSED / "links_meta.csv")
med = pd.read_csv(PROCESSED / "links_rsl_1min.csv.gz", usecols=list(meta.col)).median()
meta["snr_dry"] = meta.col.map(med) - meta.band.map(NOISE)
LR = load_link_rain(["time_utc", "col", "band", "A_dB", "miss_frac", "rain_mmh"])
LR = LR[LR.band.isin(["68-69GHz", "5GHz"]) & LR.rain_mmh.notna() & LR.miss_frac.notna()]
LR["t"] = pd.to_datetime(LR.pop("time_utc"))
LR["snr_dry"] = LR.col.map(meta.set_index("col").snr_dry).astype("float32")
LR["g"] = np.where(LR.miss_frac > 0.5, 0.0, se(LR.snr_dry - LR.A_dB.fillna(0).clip(lower=0)) / se(LR.snr_dry))
H = LR[LR.band == "68-69GHz"][["t", "col", "g", "rain_mmh"]].copy(); H["col"] = H.col.astype(str)
F = LR[LR.band == "5GHz"][["t", "g"]].rename(columns={"g": "gF"}).sample(frac=1, random_state=1).drop_duplicates("t")
del LR
H = H.merge(F, on="t", how="left"); H["gF"] = H.gF.fillna(1.0)
onH = np.minimum(1.0, HEADROOM * H.g.values)
for S in S_LIST:
    H[f"loss_{S}"] = 1 - (onH + np.minimum(1 - onH, S * H.gF.values))
H["event"] = -1
for i, r in ev.iterrows():
    H.loc[(H.t >= r.w0) & (H.t <= r.w1), "event"] = i
He = H[H.event >= 0]
# D = fixed-duration IM: maximum 24-h accumulation for windows ending in [storm start, max(end, start + 24 h)]
#     (same definition as NOAA Atlas 14 24-h depth; review item P1-2)
H = H.sort_values(["col", "t"])
H["d24"] = H.groupby("col").rain_mmh.transform(lambda r: (r * 5 / 60).rolling(288, min_periods=230).sum())
d24_site = {}
for i, r in ev.iterrows():
    t_hi = max(r.end, r.start + pd.Timedelta("24h"))
    m = (H.t >= r.start) & (H.t <= t_hi)
    d24_site[i] = H[m].groupby("col").d24.max()
He = H[H.event >= 0]
site = He.groupby(["event", "col"]).agg(P=("rain_mmh", "max"), D_storm=("rain_mmh", lambda s: s.sum() * 5 / 60),
                                       n=("rain_mmh", "size"),
                                       **{f"EOM_{S}": (f"loss_{S}", lambda s: s.sum() * 5) for S in S_LIST}).reset_index()
site = site[site.n >= 0.8 * site.groupby("event").n.transform("max")]   # drop sites with large gaps
site["D"] = [d24_site[e].get(c, np.nan) for e, c in zip(site.event, site.col)]
site = site.dropna(subset=["D"])
site["P"] = site.P.clip(lower=0.5); site["D"] = site.D.clip(lower=0.2)
site = site.merge(ev[["hours"]], left_on="event", right_index=True)
site["long"] = (site.hours >= 12) | (site.D_storm >= 25)
site.round(3).to_csv(OUT / "site_storm_table.csv", index=False)
print(f"site-storm rows: {len(site)} ({site.event.nunique()} storms x {site.col.nunique()} sites); "
      f"corr(lnP,lnD) = {np.corrcoef(np.log(site.P), np.log(site.D))[0, 1]:.2f}")

# ============================================================ loss models, leave-one-STORM-out
rows, loeo, coefs = [], [], []
for S in S_LIST:
    y = np.log1p(site[f"EOM_{S}"].values)
    for m in MODELS:
        X = design(site.P.values, site.D.values, m)
        b, sd = ols(X, y); coefs.append(dict(S=S, model=m, coef=np.round(b, 3).tolist(), resid_sd=round(sd, 3)))
        pred = np.empty(len(y)); pmean = np.empty(len(y))
        for e in site.event.unique():
            k = site.event.values == e
            bi, sdi = ols(X[~k], y[~k])                       # coefficients AND residual sd from the training fold only
            pred[k] = X[k] @ bi
            pmean[k] = np.expm1(pred[k] + sdi ** 2 / 2)       # E[EOM] = exp(mu + sd^2/2) - 1
        d = site[["event", "col", "P", "D", "hours", "long"]].assign(S=S, model=m, obs=site[f"EOM_{S}"].values,
                                                                     pred=pmean, pred_median=np.expm1(pred))
        loeo.append(d)
        evl = d.groupby("event").agg(obs=("obs", "mean"), pred=("pred", "mean"), long=("long", "mean"))
        lg = evl.long.values > 0.5
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        ly, lp = np.log1p(evl.obs), np.log1p(evl.pred)
        r2e = 1 - ((ly - lp) ** 2).sum() / ((ly - ly.mean()) ** 2).sum()
        rows.append(dict(S=S, model=m, R2_log_site=r2, R2_log_storm=r2e, rmse_log_site=np.sqrt(np.mean((y - pred) ** 2)),
                         sum_ratio_all=evl.pred.sum() / evl.obs.sum(),
                         sum_ratio_long=evl.pred[lg].sum() / evl.obs[lg].sum(),
                         sum_ratio_short=evl.pred[~lg].sum() / evl.obs[~lg].sum(), n_long_storms=int(lg.sum())))
val = pd.DataFrame(rows); loeo = pd.concat(loeo)
val.round(3).to_csv(OUT / "loss_model_loeo_validation.csv", index=False)
pd.DataFrame(coefs).to_csv(OUT / "loss_model_coefficients.csv", index=False)
loeo.round(2).to_csv(OUT / "loss_model_loeo_predictions.csv", index=False)
print(val.round(3).to_string()); print(pd.DataFrame(coefs).to_string())

# ============================================================ station-storm P, D for the hazard
a = pd.read_csv(ASOS_CSV)
a["t"] = pd.to_datetime(a["valid(UTC)"]); a["mm"] = pd.to_numeric(a.precip, errors="coerce") * 25.4
A = a.pivot_table(index="t", columns="station", values="mm").reindex(pd.date_range("2023-10-29", "2024-07-01", freq="min"))
A15 = A.rolling(15, min_periods=12).sum() * 4
p = pd.read_csv(PROCESSED / "pws_5min.csv.gz", parse_dates=["time_utc"])
Pw = p.pivot_table(index="time_utc", columns="station", values="rainfall_amount").resample("5min").sum(min_count=1)
qc = pd.read_csv(R2 / "pws_qc.csv", index_col=0); good = list(qc.index[qc.keep])
Pw15 = Pw[good].rolling(3, min_periods=2).sum() * 4
A24 = A.rolling(1440, min_periods=1150).sum()
Pw24 = Pw[good].rolling(288, min_periods=230).sum()
st = []
for i, r in ev.iterrows():
    s, t = r.w0, r.w1
    t_hi = max(r.end, r.start + pd.Timedelta("24h"))
    for c in A.columns:
        seg = A[c][s:t]
        if seg.notna().mean() > 0.8:
            st.append(("ASOS", c, i, A15[c][s:t].max(), A24[c][r.start:t_hi].max()))
    for c in good:
        seg = Pw[c][s:t]
        if seg.notna().mean() > 0.8:
            st.append(("PWS", c, i, Pw15[c][s:t].max(), Pw24[c][r.start:t_hi].max()))
st = pd.DataFrame(st, columns=["src", "station", "event", "P", "D"]).dropna()
st = st[st.P >= P_MIN]
st.round(3).to_csv(OUT / "station_storm_table.csv", index=False)
n_st = st.station.nunique()
# effective exposure (review item P2-4): station-years actually observed, not stations x record length
cov_st = {c: A[c].notna().mean() for c in A.columns if c in set(st.station)}
cov_st.update({c: Pw[c].notna().mean() for c in good if c in set(st.station)})
yrs = sum(cov_st.values()) * T_OBS
print(f"effective station-years: {yrs:.2f} (= {sum(cov_st.values()):.2f} full stations x {T_OBS:.3f} yr)")
print(f"station-storms: {len(st)} from {n_st} stations")

atl_i = pd.read_csv(ATLAS_INTENSITY, comment="#").set_index("duration")
atl_d = pd.read_csv(ATLAS_DEPTH, comment="#").set_index("duration")
T_ARI = np.array([float(c) for c in atl_i.columns])
i15 = atl_i.loc["15-min"].values.astype(float); d24 = atl_d.loc["24-hr"].values.astype(float); d6 = atl_d.loc["6-hr"].values.astype(float)


def emp_rate(vals, grid):
    v = np.sort(vals)
    return np.array([max((v > g).sum(), 0.3) for g in grid]) / yrs


def gpd_fit(v, u):
    x = np.sort(v[v > u]) - u; n = len(x)
    a0 = x.mean(); a1 = np.sum((n - np.arange(1, n + 1)) / (n - 1) * x) / n
    return n / yrs, -(a0 / (a0 - 2 * a1) - 2), 2 * a0 * a1 / (a0 - 2 * a1)


lamuP, xiP, sgP = gpd_fit(st.P.values, U)
PG = np.logspace(np.log10(P_MIN), np.log10(400), 1500)
DG = np.logspace(np.log10(0.2), np.log10(800), 1500)
empP, empD = emp_rate(st.P.values, PG), emp_rate(st.D.values, DG)
gpdP = lamuP * np.maximum(1 + xiP * np.maximum(PG - U, 0) / sgP, 1e-12) ** (-1 / xiP)


def lam_P_grid(tail):
    lam = np.where(PG < U, empP, gpdP)
    if tail == "atlas":
        hi = PG >= i15[0]
        x_ = PG[hi]
        lam[hi] = np.exp(np.where(x_ <= i15[-1], np.interp(x_, i15, np.log(1 / T_ARI)), np.log(1 / T_ARI[-1]) - (x_ - i15[-1]) / 30))
    return np.minimum.accumulate(lam)


def lam_D_grid(tail):
    lam = empD.copy()
    if tail == "atlas":
        d_b, d_a = 40.0, d24[0]
        lb = np.interp(d_b, DG, empD)
        mid = (DG >= d_b) & (DG < d_a)
        lam[mid] = np.exp(np.interp(DG[mid], [d_b, d_a], [np.log(lb), np.log(1.0)]))
        hi = DG >= d_a
        x_ = DG[hi]
        lam[hi] = np.exp(np.where(x_ <= d24[-1], np.interp(x_, d24, np.log(1 / T_ARI)), np.log(1 / T_ARI[-1]) - (x_ - d24[-1]) / 60))
    else:  # observed-only: exponential tail beyond the largest observation
        dmax = st.D.max(); top = DG > dmax * 0.8
        sl = np.polyfit(DG[(DG > 20) & (DG < dmax * 0.8)], np.log(empD[(DG > 20) & (DG < dmax * 0.8)]), 1)
        lam[top] = np.exp(np.polyval(sl, DG[top]))
    return np.minimum.accumulate(lam)


# Gaussian copula from Spearman rho of station storms
rs = st[["P", "D"]].rank().corr().iloc[0, 1]
RHO = 2 * np.sin(np.pi * rs / 6)
hb, sdH = ols(np.column_stack([np.ones(len(ev)), np.log(site.groupby("event").P.median().reindex(ev.index).fillna(1).clip(lower=.5)),
                               np.log(site.groupby("event").D.median().reindex(ev.index).fillna(1).clip(lower=.2))]),
              np.log(ev.hours.clip(lower=0.25).values))
print(f"P GPD xi={xiP:.3f} sig={sgP:.2f}; Spearman={rs:.2f} -> rho={RHO:.2f}; lnH = {hb.round(2)}, sd {sdH:.2f}")


def simulate(n, tail="atlas", rho=RHO, rgen=rng):
    lP, lD = lam_P_grid(tail), lam_D_grid(tail)
    lam0 = lP[0]
    z1 = rgen.standard_normal(n); z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rgen.standard_normal(n)
    sP = 1 - ncdf(z1)                                            # exceedance prob within the storm population
    sD = 1 - ncdf(z2)
    P = np.interp(-np.log(np.maximum(sP * lam0, 1e-12)), -np.log(lP), PG)
    D = np.interp(-np.log(np.maximum(sD * lD[0], 1e-12)), -np.log(lD), DG)
    D = np.maximum(D, P * 0.25)                                  # a 15-min burst at rate P delivers P/4 mm
    Hh = np.exp(hb[0] + hb[1] * np.log(P) + hb[2] * np.log(D) + sdH * rgen.standard_normal(n))
    Hh = np.maximum(Hh, D / P)
    return P, D, Hh, lam0 / n


def loss_sim(P, D, Hh, b, sd, m, rgen):
    L = np.expm1(design(P, D, m) @ b + sd * rgen.standard_normal(len(P))).clip(min=0)
    return np.minimum(L, Hh * 60 + 60)                           # cannot lose more than the storm window


def fit_all(df, S):
    y = np.log1p(df[f"EOM_{S}"].values)
    return {m: ols(design(df.P.values, df.D.values, m), y) for m in MODELS}


# ---------------- hazard validation
Ps, Ds, Hs, w = simulate(N_SIM)
hz = pd.DataFrame({"ARI_yr": T_ARI, "atlas_15min_mmh": i15, "obs_P_rate_at_atlas": np.interp(i15, PG, np.where(PG < U, empP, gpdP)) * T_ARI,
                   "atlas_24h_mm": d24, "atlas_6h_mm": d6,
                   "obs_D_rate_at_atlas24": np.interp(d24, DG, empD) * T_ARI})
hz.round(3).to_csv(OUT / "hazard_marginals_vs_atlas14.csv", index=False)
jt = []
Po_, Do_, _, wo_ = simulate(N_SIM, tail="observed")
for pt, dt_ in [(10, 10), (20, 20), (20, 40), (30, 30), (10, 40)]:
    obs = ((st.P > pt) & (st.D > dt_)).sum() / yrs
    pr = wo_ * ((Po_ > pt) & (Do_ > dt_)).sum()
    pr0 = wo_ * ((Po_ > pt) & (rng.permutation(Do_) > dt_)).sum()
    jt.append(dict(P_gt=pt, D_gt=dt_, observed_per_station_yr=obs, copula_pred=pr, independence_pred=pr0))
jt = pd.DataFrame(jt); jt.round(3).to_csv(OUT / "hazard_joint_validation.csv", index=False)
print(hz.round(2).head(4).to_string()); print(jt.round(2).to_string())

# ============================================================ risk
L_GRID = np.logspace(0, np.log10(3000), 150)
risk, summ = {}, []
for S in S_LIST:
    fm = fit_all(site, S)
    for m in MODELS:
        b, sd = fm[m]
        L = loss_sim(Ps, Ds, Hs, b, sd, m, rng)
        rc = np.array([w * (L > l).sum() for l in L_GRID]); risk[(S, m)] = rc
        ev_ids = site.event.unique(); bt = []
        for _ in range(N_BOOT):
            pick = rng.choice(ev_ids, len(ev_ids))
            sb = pd.concat([site[site.event == e] for e in pick])
            bb, sdb = fit_all(sb, S)[m]
            Pb, Db, Hb, wb = simulate(20_000)
            bt.append(wb * loss_sim(Pb, Db, Hb, bb, sdb, m, rng).sum())
        q = lambda T: np.interp(-np.log(1 / T), -np.log(np.maximum(rc, 1e-12)), L_GRID)
        summ.append(dict(S=S, model=m, EAL=w * L.sum(), EAL_ci05=np.percentile(bt, 5), EAL_ci95=np.percentile(bt, 95),
                         EOM_1yr=q(1), EOM_10yr=q(10), EOM_100yr=q(100),
                         storms_per_yr_EOM_gt60=np.interp(60, L_GRID, rc), storms_per_yr_EOM_gt200=np.interp(200, L_GRID, rc)))
summ = pd.DataFrame(summ); summ.round(3).to_csv(OUT / "risk_summary_bivariate.csv", index=False)
pd.DataFrame({"EOM_min": L_GRID, **{f"{m}|S={S}": v for (S, m), v in risk.items()}}).to_csv(OUT / "risk_curves_bivariate.csv", index=False)
print(summ.round(2).to_string())

# back-test: observed-only hazard, site-level storm losses of the record (per site-year)
Po, Do, Ho, wo = simulate(N_SIM, tail="observed")
site_years = site.col.nunique() * T_OBS * (site.groupby("event").col.nunique().mean() / site.col.nunique())
btr = []
for S in [0.0, 0.5]:
    fm = fit_all(site, S); obs = site[f"EOM_{S}"].values
    for m in MODELS:
        L = loss_sim(Po, Do, Ho, *fm[m], m, rng)
        for l in [10, 30, 60, 100, 200]:
            btr.append(dict(S=S, model=m, EOM_gt=l, predicted_per_site_yr=wo * (L > l).sum(),
                            observed_per_site_yr=(obs > l).sum() / site_years, observed_site_storms=int((obs > l).sum())))
btest = pd.DataFrame(btr); btest.round(3).to_csv(OUT / "backtest_bivariate.csv", index=False)
print(btest.round(2).to_string())

# ============================================================ figures
C = {"M1 peak only": "tab:grey", "M2 peak + depth": "tab:green"}
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.7))
g0 = loeo[loeo.S == 0.0]
for m in MODELS:
    g = g0[g0.model == m].groupby("event").agg(obs=("obs", "mean"), pred=("pred", "mean"), hours=("hours", "first"))
    r = val[(val.S == 0) & (val.model == m)].iloc[0]
    ax[0].scatter(g.obs + 1, g.pred + 1, s=10 + 3 * g.hours, color=C[m], alpha=.65, edgecolor="k", lw=.3,
                  label=f"{m}: storm R2(log)={r.R2_log_storm:.2f}, long-storm pred/obs={r.sum_ratio_long:.2f}")
lim = [1, 800]
ax[0].plot(lim, lim, "k-", lw=.8); ax[0].plot(lim, [2 * v for v in lim], "k:", lw=.6); ax[0].plot(lim, [v / 2 for v in lim], "k:", lw=.6)
ax[0].set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="observed EOM + 1, storm mean over sites (min)",
          ylabel="predicted EOM + 1 (min)", title="(a) Leave-one-storm-out, S = 0 (size = duration)")
ax[0].legend(fontsize=6.5, loc="upper left"); ax[0].grid(alpha=.3, which="both")
for m in MODELS:
    g = g0[g0.model == m]
    res = np.log1p(g.pred_median) - np.log1p(g.obs)
    bins = pd.qcut(np.log(g.D), 8)
    q = (pd.DataFrame({"D": g.D, "res": res, "b": bins}).groupby("b", observed=True)
         .agg(D=("D", "median"), m=("res", "mean"), s=("res", "sem")))
    ax[1].errorbar(q.D, q.m, yerr=1.96 * q.s, fmt="o-", color=C[m], capsize=2, label=m)
ax[1].axhline(0, color="k", lw=.8)
ax[1].set(xscale="log", xlabel="site 24-h max depth D (mm)", ylabel="mean log residual, median prediction (95% CI)",
          title="(b) Bias vs depth: M1 under-predicts deep storms")
ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
vv = val.pivot(index="S", columns="model", values="sum_ratio_long")
xb = np.arange(len(vv)); wd = .38
for dx, m in [(-wd / 2, MODELS[0]), (wd / 2, MODELS[1])]:
    ax[2].bar(xb + dx, vv[m], wd, color=C[m], label=m)
    for k, S in enumerate(vv.index):
        r = val[(val.S == S) & (val.model == m)].iloc[0]
        ax[2].text(k + dx, vv.loc[S, m] + .02, f"{vv.loc[S, m]:.2f}\nR2 {r.R2_log_storm:.2f}", ha="center", fontsize=6.5)
ax[2].axhline(1, color="k", lw=.8, ls=":")
ax[2].set_xticks(xb, [f"S={S:.0%}" for S in vv.index])
ax[2].set(ylabel="long storms: sum(pred) / sum(obs)", ylim=(0, 1.5), title="(c) Long-storm bias by spare capacity (1 = unbiased)")
ax[2].legend(fontsize=7, loc="upper left"); ax[2].grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig(OUT / "fig8_bivariate_loss_validation.png", dpi=150)

fig, ax = plt.subplots(2, 3, figsize=(18, 9.6))
for src, mk in [("ASOS", "s"), ("PWS", "o")]:
    g = st[st.src == src]; ns = g.station.nunique()
    st_yrs = sum(cov_st[c] for c in g.station.unique()) * T_OBS          # effective station-years
    for a_, v in [(ax[0, 0], np.sort(g.P.values)[::-1]), (ax[0, 1], np.sort(g.D.values)[::-1])]:
        a_.plot(v, np.arange(1, len(v) + 1) / st_yrs, mk, ms=2.5, alpha=.5, label=f"observed {src}")
ax[0, 0].plot(PG, lam_P_grid("observed"), "k--", lw=1, label=f"observed-only (GPD xi={xiP:.2f})")
ax[0, 0].plot(PG, lam_P_grid("atlas"), "k-", lw=1.6, label="composite (used for risk)")
ax[0, 0].plot(i15, 1 / T_ARI, "D", color="tab:red", label="NOAA Atlas 14, 15-min")
ax[0, 0].set(xscale="log", yscale="log", xlim=(1, 300), ylim=(1e-3, 100), xlabel="P: peak 15-min intensity (mm/h)",
             ylabel="storms per station-year with value > x", title="(a) Peak marginal vs Atlas 14")
ax[0, 1].plot(DG, lam_D_grid("observed"), "k--", lw=1, label="observed-only")
ax[0, 1].plot(DG, lam_D_grid("atlas"), "k-", lw=1.6, label="composite (used for risk)")
ax[0, 1].plot(d24, 1 / T_ARI, "D", color="tab:purple", label="NOAA Atlas 14, 24-h depth")
ax[0, 1].plot(d6, 1 / T_ARI, "d", color="tab:orange", label="NOAA Atlas 14, 6-h depth")
ax[0, 1].set(xscale="log", yscale="log", xlim=(1, 600), ylim=(1e-3, 100), xlabel="D: max 24-h accumulation during storm (mm)",
             title="(b) 24-h depth marginal vs Atlas 14 (same definition)")
for a_ in ax[0, :2]:
    a_.legend(fontsize=7); a_.grid(alpha=.3, which="both")
idx = rng.choice(len(Ps), 5000)
ax[0, 2].scatter(Ps[idx], Ds[idx], s=2, color="tab:blue", alpha=.15, label=f"simulated (Gaussian copula, rho={RHO:.2f})")
ax[0, 2].scatter(st.P, st.D, s=6, color="tab:red", alpha=.5, label="observed station-storms")
pp = np.logspace(0, 2.5, 20); ax[0, 2].plot(pp, pp * .25, "k--", lw=.8, label="floor D = P x 15 min")
ax[0, 2].set(xscale="log", yscale="log", xlim=(1, 300), ylim=(0.2, 600), xlabel="P (mm/h)", ylabel="D (mm)",
             title="(c) Joint IM: observed vs simulated")
ax[0, 2].legend(fontsize=7, loc="upper left"); ax[0, 2].grid(alpha=.3, which="both")
for S, c_ in [(0.0, "tab:grey"), (0.5, "tab:green")]:
    ax[1, 0].plot(L_GRID, risk[(S, MODELS[0])], "--", color=c_, lw=1, label=f"S={S:.0%} M1 peak only")
    ax[1, 0].plot(L_GRID, risk[(S, MODELS[1])], "-", color=c_, lw=1.8, label=f"S={S:.0%} M2 peak + depth")
    o = np.sort(site[f"EOM_{S}"].values)[::-1]
    ax[1, 0].step(o, np.arange(1, len(o) + 1) / site_years, where="post", color=c_, ls=":", label=f"S={S:.0%} observed (per site-yr)")
for T in [1, 10, 100]:
    ax[1, 0].axhline(1 / T, color="k", lw=.4, ls="--"); ax[1, 0].text(1.1, 1.1 / T, f"{T}-yr", fontsize=7)
ax[1, 0].set(xscale="log", yscale="log", xlim=(1, 3000), ylim=(1e-3, 100), xlabel="storm loss EOM per site (min)",
             ylabel="annual exceedance frequency per site (1/yr)", title="(d) Risk curves (EXPLORATORY: Oct-Jun record, see assumptions)")
ax[1, 0].legend(fontsize=6.5, loc="upper right"); ax[1, 0].grid(alpha=.3, which="both")
for dx, m in [(-wd / 2, MODELS[0]), (wd / 2, MODELS[1])]:
    q = summ[summ.model == m]
    ax[1, 1].bar(np.arange(4) + dx, q.EAL, wd, color=C[m], label=m, yerr=[q.EAL - q.EAL_ci05, q.EAL_ci95 - q.EAL], capsize=3)
    for k, (_, r) in enumerate(q.iterrows()):
        ax[1, 1].text(k + dx, r.EAL_ci95 * 1.02, f"{r.EAL:.0f}", ha="center", fontsize=7)
ax[1, 1].set_xticks(np.arange(4), [f"S={S:.0%}" for S in S_LIST])
ax[1, 1].set(ylabel="EAL (EOM per site per year)", title="(e) Expected annual loss, 90% CI (bootstrap over storms)")
ax[1, 1].legend(fontsize=7); ax[1, 1].grid(alpha=.3, axis="y")
bq = btest[btest.model == MODELS[1]]
for S, c_, mk in [(0.0, "tab:grey", "o"), (0.5, "tab:green", "s")]:
    q = bq[bq.S == S]
    ax[1, 2].plot(q.observed_per_site_yr + .3, q.predicted_per_site_yr + .3, mk, color=c_, ms=7, label=f"S={S:.0%}")
    for _, r in q.iterrows():
        ax[1, 2].annotate(f">{r.EOM_gt:.0f} (n={r.observed_site_storms})", (r.observed_per_site_yr + .3, r.predicted_per_site_yr + .3),
                          fontsize=6.5, xytext=(4, -3), textcoords="offset points")
ax[1, 2].plot([.3, 80], [.3, 80], "k-", lw=.8)
ax[1, 2].plot([.3, 80], [.6, 160], "k:", lw=.6)
ax[1, 2].plot([.3, 80], [.15, 40], "k:", lw=.6)
ax[1, 2].set(xscale="log", yscale="log", xlabel="observed storms per site-yr (+0.3)", ylabel="predicted (+0.3)",
             title="(f) Back-test, observed-only model (M2), EOM thresholds in min")
ax[1, 2].legend(fontsize=7); ax[1, 2].grid(alpha=.3, which="both")
fig.tight_layout(); fig.savefig(OUT / "fig9_bivariate_hazard_risk.png", dpi=150)
print("done ->", OUT)
