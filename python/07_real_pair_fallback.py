r"""
Step 7: fallback strategies on REAL same-hop primary/backup pairs (review item P1).

Pairs (paths.PROCESSED / link_pairs.csv, usable = same direction, both links >= 80 % coverage,
backup dry noise <= 3 dB):  CML1 68.0 GHz -> 5.77 GHz (2.18 km), CML22 69.1 GHz -> 5.71 GHz (1.85 km).

1-min measured RSL of both links of each pair:
  baseline  72-h rolling median of known-dry minutes (weather state from step 2; unknown never dry)
  states    network-wide logging gap -> excluded;  RSL missing otherwise -> "no RSL" (proxy outage);
            RSL present but no baseline -> excluded
  capacity  attenuated Shannon on SNR = SNR_dry - A (same mapping as steps 4-6)
Strategies S1 / S2 (failover, hold) / S3 (load-aware, decision lag) simulated minute by minute in
every warm storm window (+-30 min).

Outputs -> paths.RESULTS / 07_real_pair_fallback
  fig10_real_pair_sweep.png       spare-capacity sweep, same-hop vs random backup
  fig11_real_pair_recovery.png    recovery curves, both pairs, heaviest storm
  fig12_sensitivity.png           headroom / noise floor / detection delay / hold time

Run:  python python/07_real_pair_fallback.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from paths import PROCESSED, R2, result_dir
from common import load_link_rain

OUT = result_dir("07_real_pair_fallback")
rng = np.random.default_rng(401)

BASE = dict(headroom=1.5, noise_shift=0.0, delay=1, hold=10)       # delay / hold in minutes
NOISE = {"5GHz": -88.0, "68-69GHz": -74.0}
S_GRID = np.round(np.arange(0, 1.51, 0.05), 2)
TARGET = 0.95


def se(snr_db):
    snr_db = np.asarray(snr_db, float)
    return np.where(snr_db < -10, 0.0, np.minimum(0.6 * np.log2(1 + 10 ** (snr_db / 10)), 4.4))


pairs = pd.read_csv(PROCESSED / "link_pairs.csv")
use = pairs[pairs.usable].reset_index(drop=True)
meta = pd.read_csv(PROCESSED / "links_meta.csv").set_index("col")
print(use[["pair_id", "high", "low", "length_m"]].to_string())

L = pd.read_csv(PROCESSED / "links_rsl_1min.csv.gz", parse_dates=["time_utc"]).set_index("time_utc")
gap = L.isna().mean(axis=1) > 0.5                                    # logging artefact
ws = pd.read_csv(R2 / "weather_state_5min.csv.gz", index_col=0, parse_dates=True)
wet5 = ws.state.eq("wet").astype(float).where(ws.state != "unknown")
known_dry = wet5.rolling(24, min_periods=24).sum().eq(0)            # 2 h known dry
dry1 = known_dry.reindex(L.index, method="ffill").fillna(False).astype(bool)

# random backup pool: all usable-quality 5 GHz links NOT in the same hop
cov = L.notna().mean()
pool = [c for c in meta.index[meta.band == "5GHz"] if cov[c] >= 0.8]

ev = pd.read_csv(R2 / "rain_events.csv", parse_dates=["start", "end"])
ev = ev[~ev.likely_snow_or_mixed].reset_index(drop=True)
win = pd.Series(False, index=L.index)
ev_id = pd.Series(-1, index=L.index)
for i, r in ev.iterrows():
    m = (L.index >= r.start - pd.Timedelta("30min")) & (L.index <= r.end + pd.Timedelta("30min"))
    win[m] = True; ev_id[m] = i

# rain at each hop: 3 nearest QC'd PWS 15-min rate from step 2 (5-min) -> 1-min
LRr = load_link_rain(["time_utc", "col", "rain_mmh"])


_BASE = {}


def baseline(c):
    """72-h rolling median of known-dry RSL, computed on 5-min means and held for each minute."""
    if c not in _BASE:
        r5 = L[c].where(dry1).resample("5min").mean()
        _BASE[c] = r5.rolling("72h", min_periods=216).median().ffill().reindex(L.index, method="ffill")
    return _BASE[c]


def link_state(c, band, noise_shift=0.0):
    rsl = L[c]
    base = baseline(c)
    A = (base - rsl).clip(lower=0)
    snr0 = base.median() - NOISE[band] + noise_shift
    g = pd.Series(se(snr0 - A) / se(snr0), index=L.index)
    g[rsl.isna()] = 0.0                                              # no RSL -> proxy outage
    valid = ~gap & base.notna()
    return g.where(valid), A.where(valid & rsl.notna())


def simulate(gH, gF, S, p):
    """gH, gF arrays (1-min, NaN = excluded). Returns served share per minute for S1, S2, S3."""
    capH = p["headroom"] * np.nan_to_num(gH, nan=1 / p["headroom"] * 2)
    capF = S * np.nan_to_num(gF, nan=1.0)
    d = int(p["delay"])
    meas = np.r_[np.full(d, p["headroom"]), capH[:len(capH) - d]] if d > 0 else capH
    s1 = np.minimum(1, capH)
    assign = np.minimum(1, meas)
    s3 = np.minimum(assign, capH) + np.minimum(1 - assign, capF)
    s2 = np.empty_like(s1); onF, h = False, 0
    for i in range(len(s1)):
        if not onF and meas[i] < 1:
            onF, h = True, 0
        elif onF:
            h = h + 1 if meas[i] >= 1 else 0
            if meas[i] >= 1 and h >= p["hold"]:
                onF = False
        s2[i] = min(1.0, capF[i]) if onF else s1[i]
    return s1, s2, s3


def run_pair(hc, lc, S, p, gF_override=None):
    gH, AH = link_state(hc, "68-69GHz", p["noise_shift"])
    gF, AF = link_state(lc, "5GHz", p["noise_shift"])
    if gF_override is not None:
        gF = gF_override
    sel = win.values & gH.notna().values
    out = {}
    s = simulate(gH.values[sel], gF.values[sel], S, p)
    eid = ev_id.values[sel]
    for k, v in zip(["S1", "S2", "S3"], s):
        out[k] = pd.DataFrame({"event": eid, "served": v})
    return out, sel


# ---------------------------------------------------------------- sweep (same-hop vs random backup)
rows = []
for _, pr in use.iterrows():
    gH_, _ = link_state(pr.high, "68-69GHz")
    gF_same, _ = link_state(pr.low, "5GHz")
    # random backup: at each minute a 5 GHz link from another hop (same minute -> same storm)
    others = [c for c in pool if not c.startswith(f"cml{pr.cml_id}_")]
    G = pd.concat({c: link_state(c, "5GHz")[0] for c in others}, axis=1)
    pick = rng.integers(0, G.shape[1], len(G))
    g_rand = pd.Series(G.values[np.arange(len(G)), pick], index=G.index)
    sel = win.values & gH_.notna().values
    for label, gF in [("same-hop backup (measured)", gF_same), ("random other-hop backup (synthetic)", g_rand)]:
        for S in S_GRID:
            s1, s2, s3 = simulate(gH_.values[sel], gF.values[sel], S, BASE)
            for k, v in zip(["S1", "S2", "S3"], [s1, s2, s3]):
                rows.append(dict(pair=pr.pair_id, backup=label, S=S, strategy=k, mean_service=v.mean(),
                                 share_minutes_meeting_target=(v >= TARGET).mean(), EOM_total=(1 - v).sum(),
                                 n_minutes=len(v)))
sw = pd.DataFrame(rows)
sw.round(4).to_csv(OUT / "real_pair_sweep.csv", index=False)
n_events = ev.shape[0]
sw["EOM_per_storm"] = sw.EOM_total / n_events

# per-storm table at the base case (S = 0.5)
per = []
for _, pr in use.iterrows():
    gH_, _ = link_state(pr.high, "68-69GHz"); gF_, _ = link_state(pr.low, "5GHz")
    sel = win.values & gH_.notna().values
    s = simulate(gH_.values[sel], gF_.values[sel], 0.5, BASE)
    df = pd.DataFrame({"event": ev_id.values[sel], "S1": s[0], "S2": s[1], "S3": s[2]})
    agg = df.groupby("event").agg(**{f"EOM_{k}": (k, lambda x: (1 - x).sum()) for k in ["S1", "S2", "S3"]},
                                  minutes=("S1", "size")).reset_index()
    agg["pair"] = pr.pair_id
    per.append(agg.merge(ev[["start", "hours", "depth_mm", "peak15_mmh"]], left_on="event", right_index=True))
per = pd.concat(per); per.round(2).to_csv(OUT / "real_pair_storm_losses_S50.csv", index=False)

# ---------------------------------------------------------------- sensitivity (S = 0.5, pooled pairs, S3 and S2)
sens_cases = [("headroom", [1.2, 1.5, 2.0, 3.0]), ("noise_shift", [-3.0, 0.0, 3.0]),
              ("delay", [0, 1, 5]), ("hold", [0, 10, 30])]
sens = []
for par, vals in sens_cases:
    for v in vals:
        p = dict(BASE); p[par] = v
        tot = {"S1": 0.0, "S2": 0.0, "S3": 0.0}; n = 0
        for _, pr in use.iterrows():
            gH_, _ = link_state(pr.high, "68-69GHz", p["noise_shift"]); gF_, _ = link_state(pr.low, "5GHz", p["noise_shift"])
            sel = win.values & gH_.notna().values
            s = simulate(gH_.values[sel], gF_.values[sel], 0.5, p)
            for k, x in zip(["S1", "S2", "S3"], s):
                tot[k] += (1 - x).sum()
            n += 1
        for k in tot:
            sens.append(dict(parameter=par, value=v, strategy=k, EOM_per_storm_per_pair=tot[k] / (n * n_events)))
sens = pd.DataFrame(sens); sens.round(3).to_csv(OUT / "sensitivity_S50.csv", index=False)
print(sens.pivot_table(index=["parameter", "value"], columns="strategy", values="EOM_per_storm_per_pair").round(2))

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
cols = {"S1": "tab:grey", "S2": "tab:orange", "S3": "tab:green"}
for a_, pr in zip(ax, use.pair_id):
    for (bk, st), g in sw[sw.pair == pr].groupby(["backup", "strategy"]):
        if st == "S1" and "random" in bk:
            continue
        ls = "-" if "same" in bk else "--"
        a_.plot(g.S * 100, g.mean_service * 100, ls, color=cols[st], lw=1.8 if st == "S3" else 1.2,
                label=f"{st} {'no redistribution' if st == 'S1' else ''}{bk if st != 'S1' else ''}")
    a_.axhline(TARGET * 100, color="k", lw=.7, ls=":")
    a_.set(title=f"{pr} (measured RSL, {n_events} storms)", xlabel="spare backup capacity S (% of primary users' basic demand)",
           ylabel="mean share of users at minimum service during storms (%)")
    a_.legend(fontsize=6.5, loc="lower right"); a_.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "fig10_real_pair_sweep.png", dpi=150)

e = ev.sort_values("peak15_mmh").iloc[-1]
w0, w1 = e.start - pd.Timedelta("45min"), e.end + pd.Timedelta("90min")
fig, ax = plt.subplots(3, 2, figsize=(14, 7.5), sharex=True, gridspec_kw=dict(height_ratios=[.8, 1, 1.6]))
met = []
for j, (_, pr) in enumerate(use.iterrows()):
    gH_, AH = link_state(pr.high, "68-69GHz"); gF_, AF = link_state(pr.low, "5GHz")
    sl = slice(w0, w1)
    s = simulate(gH_[sl].values, gF_[sl].values, 0.5, BASE)
    t = gH_[sl].index
    rr = LRr[LRr.col.astype(str) == pr.high].set_index("time_utc").rain_mmh.loc[w0:w1]
    ax[0, j].bar(rr.index, rr.values, width=0.003, color="tab:blue"); ax[0, j].set(ylabel="rain (mm/h)", title=pr.pair_id)
    ax[1, j].plot(t, AH[sl], label=f"{meta.loc[pr.high].frequency / 1000:.1f} GHz primary")
    ax[1, j].plot(t, AF[sl], label=f"{meta.loc[pr.low].frequency / 1000:.2f} GHz backup")
    nr = gH_[sl] == 0
    ax[1, j].scatter(t[nr.values], np.zeros(nr.sum()) + 1, marker="|", color="red", s=40, label="primary: no RSL")
    ax[1, j].set(ylabel="extra loss (dB)"); ax[1, j].legend(fontsize=7)
    for k, v in zip(["S1", "S2", "S3"], s):
        ax[2, j].plot(t, v * 100, color=cols[k], lw=1.8 if k == "S3" else 1.2,
                      label=f"{k}: {(v < TARGET).sum()} min < 95 %, EOM {(1 - v).sum():.0f}")
        met.append(dict(pair=pr.pair_id, strategy=k, min_service=v.min(), minutes_below_95=int((v < TARGET).sum()), EOM=(1 - v).sum()))
    ax[2, j].axhline(TARGET * 100, color="k", lw=.7, ls=":")
    ax[2, j].set(ylabel="users at min. service (%)", xlabel=f"UTC, {e.start:%Y-%m-%d}")
    ax[2, j].legend(fontsize=7, loc="lower right")
for a_ in ax.ravel():
    a_.grid(alpha=.3)
    a_.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
fig.suptitle(f"Recovery curves on real primary/backup pairs, storm {e.start:%Y-%m-%d} "
             f"(peak {e.peak15_mmh:.0f} mm/h), S = 50 %, headroom {BASE['headroom']}x")
fig.tight_layout(); fig.savefig(OUT / "fig11_real_pair_recovery.png", dpi=150)
pd.DataFrame(met).round(3).to_csv(OUT / "real_pair_recovery_metrics.csv", index=False)

fig, ax = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
for a_, (par, vals) in zip(ax, sens_cases):
    q = sens[sens.parameter == par]
    for k in ["S1", "S2", "S3"]:
        z = q[q.strategy == k]
        a_.plot(z.value, z.EOM_per_storm_per_pair, "o-", color=cols[k], label=k)
    a_.axvline(BASE[par], color="k", lw=.6, ls=":")
    a_.set(xlabel=par.replace("_", " ") + (" (dB)" if par == "noise_shift" else " (min)" if par in ("delay", "hold") else " (x)"),
           title=f"sensitivity: {par}"); a_.grid(alpha=.3)
ax[0].set_ylabel("lost user-minutes per storm per pair (S = 50 %)"); ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "fig12_sensitivity.png", dpi=150)
print(sw[(sw.S.isin([0, 0.25, 0.5, 1.0])) & (sw.strategy != "S1")].pivot_table(
    index=["pair", "backup", "strategy"], columns="S", values="mean_service").round(3))
print(sw[(sw.strategy == "S1") & (sw.S == 0)][["pair", "backup", "mean_service"]].drop_duplicates("pair"))
print(pd.DataFrame(met).round(3).to_string())
print("done ->", OUT)
