r"""
Step 5: fragility curves and out-of-sample validation (68-69 GHz -> 5 GHz, OpenMesh NYC).

  fragility      P(DS >= k | R): DS1 degraded (high band cannot carry basic demand), DS2 no-RSL proxy
                 outage. Lognormal with a dry-weather floor p0, fit on data BEFORE 2024-04-01 and scored
                 on data AFTER (held out). Benchmarks: climatology and ITU-R P.838 physics.
  vulnerability  V_S(R) = expected share of users below minimum service at rain rate R with spare
                 backup capacity S (strategy S3); validated leave-one-storm-out on storm loss
                 (EOM = lost user-minutes per site per storm).
Annual frequency and risk are computed in step 6 (fixed-duration intensity measure, effective exposure).

Outputs -> paths.RESULTS / 05_validation_risk
    fragility_parameters.csv, validation_brier.csv, validation_reliability_test.csv,
    validation_event_loss_loeo.csv, validation_event_loss_summary.csv, fig4, fig5

Run:  python python/05_validation_and_risk.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paths import PROCESSED, R2, result_dir
from common import load_link_rain

OUT = result_dir("05_validation_risk")
rng = np.random.default_rng(401)

HEADROOM = 1.5
S_LIST = [0.0, 0.25, 0.5, 1.0]
SPLIT = "2024-04-01"
NOISE = {"5GHz": -88.0, "24GHz": -84.0, "60GHz": -74.0, "68-69GHz": -74.0}
V_EDGES = [0, 1e-9, 1, 2, 5, 10, 20, 40, np.inf]


def se(snr_db):
    snr_db = np.asarray(snr_db, float)
    return np.where(snr_db < -10, 0.0, np.minimum(0.6 * np.log2(1 + 10 ** (snr_db / 10)), 4.4))


def ncdf(x):  # standard normal CDF (Abramowitz-Stegun 7.1.26 erf, |err| < 1.5e-7)
    z = np.abs(x) / np.sqrt(2)
    t = 1 / (1 + 0.3275911 * z)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-z * z)
    return 0.5 * (1 + np.sign(x) * y)


# ===================================================================== data
meta = pd.read_csv(PROCESSED / "links_meta.csv")
med = pd.read_csv(PROCESSED / "links_rsl_1min.csv.gz", usecols=list(meta.col)).median()
meta["snr_dry"] = meta.col.map(med) - meta.band.map(NOISE)
LR = load_link_rain(["time_utc", "col", "band", "A_dB", "A_itu_dB", "miss_frac", "rain_mmh", "itu_k", "itu_a"])
LR = LR[LR.band.isin(["68-69GHz", "60GHz", "5GHz"]) & LR.rain_mmh.notna()]
LR["t"] = pd.to_datetime(LR.pop("time_utc"))
LR["snr_dry"] = LR.col.map(meta.set_index("col").snr_dry).astype("float32")
LR["g"] = np.where(LR.miss_frac > 0.5, 0.0, se(LR.snr_dry - LR.A_dB.fillna(0).clip(lower=0)) / se(LR.snr_dry))
LR = LR[LR.g.notna()].copy()
LR["DS1"] = (HEADROOM * LR.g < 1).astype(int)
LR["DS2"] = (LR.g == 0).astype(int)
g_itu = se(LR.snr_dry - LR.A_itu_dB) / se(LR.snr_dry)
LR["ITU_DS1"] = (HEADROOM * g_itu < 1).astype(int)
LR["ITU_DS2"] = ((LR.snr_dry - LR.A_itu_dB) < -10).astype(int)
LR["day"] = LR.t.dt.floor("D")
LR["train"] = LR.t < SPLIT

H = LR[LR.band == "68-69GHz"].copy()
H["col"] = H.col.astype(str)
F = LR[LR.band == "5GHz"][["t", "g"]].rename(columns={"g": "gF"}).sample(frac=1, random_state=1).drop_duplicates("t")
H = H.merge(F, on="t", how="left")
H["gF"] = H.gF.fillna(1.0)
H["onH"] = np.minimum(1.0, HEADROOM * H.g)
for S in S_LIST:
    H[f"loss_{S}"] = 1 - (H.onH + np.minimum(1 - H.onH, S * H.gF))

# ===================================================================== A1 fragility + validation
TH = np.logspace(-0.5, 3.5, 161)
BE = np.linspace(0.2, 3.0, 57)


def fit_frag(R, y):
    p0 = y[R == 0].mean()
    w = R > 0
    df = pd.DataFrame({"r": np.round(R[w], 1).clip(0.1), "y": y[w]}).groupby("r").y.agg(["sum", "count"])
    r, k, n = df.index.values, df["sum"].values, df["count"].values
    lr = np.log(r)[None, None, :]
    p = p0 + (1 - p0) * ncdf((lr - np.log(TH)[:, None, None]) / BE[None, :, None])
    p = np.clip(p, 1e-9, 1 - 1e-9)
    ll = (k * np.log(p) + (n - k) * np.log(1 - p)).sum(-1)
    i, j = np.unravel_index(np.argmax(ll), ll.shape)
    return dict(p0=p0, theta=TH[i], beta=BE[j])


def frag_p(R, f):
    R = np.asarray(R, float)
    out = np.full(R.shape, f["p0"])
    w = R > 0
    out[w] = f["p0"] + (1 - f["p0"]) * ncdf(np.log(R[w] / f["theta"]) / f["beta"])
    return out


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


frag_rows, val_rows, rel_rows, fits = [], [], [], {}
for band in ["68-69GHz", "60GHz", "5GHz"]:
    B = LR[LR.band == band]
    tr, te = B[B.train], B[~B.train]
    for ds in ["DS1", "DS2"]:
        f = fit_frag(tr.rain_mmh.values, tr[ds].values)
        f_all = fit_frag(B.rain_mmh.values, B[ds].values)
        # bootstrap over days for parameter CI (training set)
        days = tr.day.unique(); grp = {d: g for d, g in tr.groupby("day")}
        bt = []
        for _ in range(60):
            pick = pd.concat([grp[d] for d in rng.choice(days, len(days))])
            bt.append(fit_frag(pick.rain_mmh.values, pick[ds].values))
        bt = pd.DataFrame(bt)
        fits[(band, ds)] = (f, bt)
        frag_rows.append(dict(band=band, DS=ds, p0=f["p0"], theta_mmh=f["theta"], beta=f["beta"],
                              theta_ci=f"{bt.theta.quantile(.05):.1f}-{bt.theta.quantile(.95):.1f}",
                              beta_ci=f"{bt.beta.quantile(.05):.2f}-{bt.beta.quantile(.95):.2f}",
                              theta_all_data=f_all["theta"], beta_all_data=f_all["beta"]))
        # out-of-sample scoring on wet test windows
        w = te[te.rain_mmh > 0]
        y = w[ds].values
        p_frag = frag_p(w.rain_mmh.values, f)
        p_clim = np.full(len(w), tr.loc[tr.rain_mmh > 0, ds].mean())
        p_itu = w[f"ITU_{ds}"].values.astype(float)
        bs = {k: np.mean((p - y) ** 2) for k, p in [("fragility", p_frag), ("climatology", p_clim), ("ITU-R P.838", p_itu)]}
        val_rows.append(dict(band=band, DS=ds, n_test_wet=len(w), test_days=w.day.nunique(),
                             brier_fragility=bs["fragility"], brier_climatology=bs["climatology"], brier_itu=bs["ITU-R P.838"],
                             BSS_vs_clim=1 - bs["fragility"] / bs["climatology"],
                             BSS_itu_vs_clim=1 - bs["ITU-R P.838"] / bs["climatology"]))
        te2 = te.assign(rb=pd.cut(te.rain_mmh, V_EDGES, right=False, include_lowest=True))
        for rb, g in te2.groupby("rb", observed=True):
            lo, hi = wilson(g[ds].sum(), len(g))
            rel_rows.append(dict(band=band, DS=ds, bin=str(rb), r_med=g.rain_mmh.median(), n=len(g), days=g.day.nunique(),
                                 obs=g[ds].mean(), obs_lo=lo, obs_hi=hi, pred=frag_p(g.rain_mmh.values, f).mean(),
                                 pred_itu=g[f"ITU_{ds}"].mean()))
frag = pd.DataFrame(frag_rows); val = pd.DataFrame(val_rows); rel = pd.DataFrame(rel_rows)
frag.round(3).to_csv(OUT / "fragility_parameters.csv", index=False)
val.round(4).to_csv(OUT / "validation_brier.csv", index=False)
rel.round(4).to_csv(OUT / "validation_reliability_test.csv", index=False)
print(frag.round(3).to_string()); print(val.round(3).to_string())

# --- Fig 4 fragility curves + held-out observations
fig, ax = plt.subplots(1, 3, figsize=(14, 4.3), sharey=True)
rr = np.logspace(-1, np.log10(250), 200)
for a, band in zip(ax, ["68-69GHz", "60GHz", "5GHz"]):
    for ds, c in [("DS1", "tab:orange"), ("DS2", "tab:red")]:
        f, bt = fits[(band, ds)]
        curves = np.array([frag_p(rr, b) for _, b in bt.iterrows()])
        a.fill_between(rr, np.percentile(curves, 5, 0), np.percentile(curves, 95, 0), color=c, alpha=.15)
        lab = "DS1 degraded" if ds == "DS1" else "DS2 outage"
        a.plot(rr, frag_p(rr, f), color=c, label=f"{lab}: fit on train (theta={f['theta']:.0f}, beta={f['beta']:.2f})")
        q = rel[(rel.band == band) & (rel.DS == ds) & (rel.r_med > 0)]
        a.errorbar(q.r_med, q.obs, yerr=[q.obs - q.obs_lo, q.obs_hi - q.obs], fmt="o", color=c, ms=4, capsize=2,
                   label=f"{ds} observed, held-out Apr-Jun (95% CI)")
        q = rel[(rel.band == band) & (rel.DS == ds)]
        if band != "5GHz":
            a.plot(q.r_med.clip(lower=.1), q.pred_itu, "x:", color=c, ms=5, lw=.8, label=f"{ds} ITU-R P.838 physics")
    a.set_xscale("log"); a.set(title=band, xlabel="15-min rain rate at link (mm/h)"); a.grid(alpha=.3)
    a.legend(fontsize=6.3, loc="upper left")
ax[0].set_ylabel("P(damage state reached)")
fig.suptitle(f"Fragility curves: lognormal fit on Oct-Mar, validated on held-out Apr-Jun "
             f"(headroom {HEADROOM}x); band = 90% bootstrap", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "fig4_fragility_validation.png", dpi=150)

# ===================================================================== A2 vulnerability + LOEO event validation
H["vb"] = pd.cut(H.rain_mmh, V_EDGES, right=False, labels=False)
ev = pd.read_csv(R2 / "rain_events.csv", parse_dates=["start", "end"])
ev = ev[~ev.likely_snow_or_mixed].reset_index(drop=True)
H["event"] = -1
for i, r in ev.iterrows():
    m = (H.t >= r.start - pd.Timedelta("30min")) & (H.t <= r.end + pd.Timedelta("30min"))
    H.loc[m, "event"] = i


def vtable(df, S):
    s = df.groupby("vb")[f"loss_{S}"].agg(["sum", "count"]).reindex(range(len(V_EDGES) - 1)).fillna(0)
    return s


def v_from(s):
    v = (s["sum"] / s["count"].replace(0, np.nan)).ffill().fillna(0).values
    v[1:] = np.maximum.accumulate(v[1:])              # monotone in rain rate (wet bins)
    return v


n_links = H.col.nunique()
loeo = []
for S in S_LIST:
    tot = vtable(H, S)
    for i in range(len(ev)):
        e = H[H.event == i]
        if e.empty:
            continue
        v = v_from(tot - vtable(e, S))
        obs = e.groupby("col")[f"loss_{S}"].sum().mean() * 5
        pred = e.assign(v=v[e.vb.values]).groupby("col").v.sum().mean() * 5
        loeo.append(dict(S=S, event=i, start=ev.start[i], peak15=ev.peak15_mmh[i], obs_EOM=obs, pred_EOM=pred))
loeo = pd.DataFrame(loeo)
loeo.round(2).to_csv(OUT / "validation_event_loss_loeo.csv", index=False)
vs = []
for S, g in loeo.groupby("S"):
    r2 = 1 - ((g.obs_EOM - g.pred_EOM) ** 2).sum() / ((g.obs_EOM - g.obs_EOM.mean()) ** 2).sum()
    rho = g[["obs_EOM", "pred_EOM"]].rank().corr().iloc[0, 1]
    within2 = ((g.pred_EOM / g.obs_EOM.clip(lower=1e-6)).between(.5, 2)).mean()
    vs.append(dict(S=S, n_events=len(g), R2=r2, spearman=rho, total_pred_over_obs=g.pred_EOM.sum() / g.obs_EOM.sum(),
                   share_within_factor2=within2))
vs = pd.DataFrame(vs); vs.round(3).to_csv(OUT / "validation_event_loss_summary.csv", index=False)
print(vs.round(3).to_string())

fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for S, c in zip(S_LIST, ["tab:grey", "tab:blue", "tab:green", "tab:purple"]):
    g = loeo[loeo.S == S]; r = vs[vs.S == S].iloc[0]
    ax[0].scatter(g.obs_EOM, g.pred_EOM, s=14, color=c, alpha=.8,
                  label=f"S={S:.0%}: R2={r.R2:.2f}, rho={r.spearman:.2f}, sum ratio={r.total_pred_over_obs:.2f}")
lim = [0.5, max(loeo.obs_EOM.max(), loeo.pred_EOM.max()) * 1.3]
ax[0].plot(lim, lim, "k-", lw=.8); ax[0].plot(lim, [2 * x for x in lim], "k:", lw=.6); ax[0].plot(lim, [x / 2 for x in lim], "k:", lw=.6)
ax[0].set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="observed EOM per site (min)", ylabel="predicted EOM per site (min)",
          title="(a) Event loss, leave-one-event-out (44 warm events)")
ax[0].legend(fontsize=6.5, loc="upper left"); ax[0].grid(alpha=.3)
v_tab = val.copy()
xb = np.arange(len(v_tab)); wdt = .27
ax[1].bar(xb - wdt, v_tab.brier_climatology, wdt, label="climatology", color="lightgrey")
ax[1].bar(xb, v_tab.brier_itu, wdt, label="ITU-R P.838 physics", color="tab:blue")
ax[1].bar(xb + wdt, v_tab.brier_fragility, wdt, label="fitted fragility", color="tab:orange")
ax[1].set_xticks(xb, [f"{b}\n{d}" for b, d in zip(v_tab.band, v_tab.DS)], fontsize=7)
ax[1].set(ylabel="Brier score on held-out wet windows (lower = better)", title="(b) Out-of-sample skill, test Apr-Jun 2024")
ax[1].legend(fontsize=7); ax[1].grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig(OUT / "fig5_validation.png", dpi=150)
print("done ->", OUT)
