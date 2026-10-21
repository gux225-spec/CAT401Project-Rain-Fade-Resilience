r"""
Step 4 / Figures 2-3: traffic redistribution to a fallback band.

System (abstracted dual-band site):
  * High band H = 68-69 GHz (mmWave proxy). Its capacity in every 5-min window comes from
    MEASURED OpenMesh RSL -> SNR -> attenuated-Shannon spectral efficiency (3GPP TR 36.942).
    RSL missing (outside network-wide logging gaps) = outage, capacity 0.
  * Fallback band F = 5 GHz (sub-6 proxy), same mapping, measured at the same time.
  * N_H users on H, each needs Q_MIN ("basic service"). In dry weather H carries
    HEADROOM x the total basic demand. Units are normalised: 1.0 = N_H * Q_MIN.
  * Spare fallback capacity S (same units) = what F can give after its own users.

Strategies:
  S1 no redistribution     users stay on H; H serves min(1, HEADROOM*g_H) of them.
  S2 link-level failover   if H cannot serve everyone, ALL users go to F (H capacity unused);
                           return to H only after H is healthy for HOLD_MIN minutes.
  S3 load-aware partial    keep as many users on H as it can serve, move only the overflow
                           to F; decisions use the previous minute's measurement.

Outputs -> paths.RESULTS / 04_fallback_model

Run:  python python/04_fallback_model.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paths import PROCESSED, result_dir
from common import load_link_rain

OUT = result_dir("04_fallback_model")
rng = np.random.default_rng(401)

# ------------------------------------------------------------------ assumptions
HEADROOM = 1.5            # dry H capacity / basic demand of its users (H loaded at 67%)
S_EVENT = 0.5             # spare fallback capacity used for the recovery curve
HOLD_MIN = 10             # S2 return-to-primary hold time (min)
NOISE = {"5GHz": -88.0, "24GHz": -84.0, "60GHz": -74.0, "68-69GHz": -74.0}  # dBm
#   thermal noise -174 dBm/Hz + 10log10(B) + NF 7 dB; B = 2 GHz (60/70 GHz radios), 80 MHz (5 GHz), 250 MHz (24 GHz)
S_GRID = np.round(np.arange(0, 2.01, 0.05), 2)
BINS = [(2, 5), (5, 10), (10, 20), (20, 300)]
BIN_LAB = ["2-5 mm/h", "5-10 mm/h", "10-20 mm/h", ">=20 mm/h"]
TARGET = 0.95
N_BOOT = 300


def se(snr_db):
    snr_db = np.asarray(snr_db, float)
    s = 0.6 * np.log2(1 + 10 ** (snr_db / 10))
    return np.where(snr_db < -10, 0.0, np.minimum(s, 4.4))


meta = pd.read_csv(PROCESSED / "links_meta.csv")
Lraw = pd.read_csv(PROCESSED / "links_rsl_1min.csv.gz", parse_dates=["time_utc"]).set_index("time_utc")
meta["snr_dry"] = meta.col.map(Lraw.median()) - meta.band.map(NOISE)

# ------------------------------------------------------------------ Fig 2: Monte Carlo sweep
LR = load_link_rain(["time_utc", "col", "band", "A_dB", "miss_frac", "rain_mmh"])
LR["col"] = LR.col.astype(str); LR["band"] = LR.band.astype(str)
LR = LR.merge(meta[["col", "snr_dry"]], on="col").dropna(subset=["rain_mmh"])
LR["g"] = np.where(LR.miss_frac > 0.5, 0.0,
                   se(LR.snr_dry - LR.A_dB.fillna(0).clip(lower=0)) / se(LR.snr_dry))
LR = LR[LR.g.notna()]
H = LR[LR.band == "68-69GHz"].copy()
F = LR[LR.band == "5GHz"][["time_utc", "g"]].rename(columns={"g": "gF"})
# pair every H window with one random 5 GHz window at the SAME time (same storm)
F = F.sample(frac=1, random_state=1).drop_duplicates("time_utc")
H = H.merge(F, on="time_utc", how="left")
H["gF"] = H.gF.fillna(1.0)
H["day"] = H.time_utc.dt.floor("D")


def served(gH, gF, S, strategy):
    onH = np.minimum(1.0, HEADROOM * gH)
    if strategy == "S1":
        return onH
    if strategy == "S3":
        return onH + np.minimum(1 - onH, S * gF)
    # S2 binary failover
    return np.where(onH >= 1.0, 1.0, np.minimum(1.0, S * gF))


rows = []
curves = {}
for (lo, hi), lab in zip(BINS, BIN_LAB):
    sub = H[(H.rain_mmh >= lo) & (H.rain_mmh < hi)]
    days = sub.day.unique()
    by_day = {d: g for d, g in sub.groupby("day")}
    for strat in ["S3", "S2"]:
        mean = np.array([served(sub.g.values, sub.gF.values, S, strat).mean() for S in S_GRID])
        # cluster bootstrap over rain days (windows on one day are not independent)
        boot = np.empty((N_BOOT, len(S_GRID)))
        for b in range(N_BOOT):
            pick = pd.concat([by_day[d] for d in rng.choice(days, len(days))])
            boot[b] = [served(pick.g.values, pick.gF.values, S, strat).mean() for S in S_GRID]
        lo95, hi95 = np.percentile(boot, [2.5, 97.5], axis=0)
        curves[(lab, strat)] = (mean, lo95, hi95)
        need = S_GRID[np.argmax(mean >= TARGET)] if (mean >= TARGET).any() else np.nan
        for S, m, a, c in zip(S_GRID, mean, lo95, hi95):
            rows.append(dict(rain_bin=lab, strategy=strat, spare_S=S, served=m, ci_lo=a, ci_hi=c,
                             n_windows=len(sub), n_days=len(days)))
        print(f"{lab:11s} {strat}: n={len(sub):5d} days={len(days):2d}  served@S=0 {mean[0]:.3f}  "
              f"@S=0.5 {mean[S_GRID == 0.5][0]:.3f}  S for {TARGET:.0%}: {need}")
sweep = pd.DataFrame(rows)
sweep.round(4).to_csv(OUT / "fig2_spare_capacity_sweep.csv", index=False)

fig, ax = plt.subplots(figsize=(7.5, 4.8))
colors = ["tab:green", "tab:blue", "tab:orange", "tab:red"]
for lab, c in zip(BIN_LAB, colors):
    m, a, b = curves[(lab, "S3")]
    ax.plot(S_GRID * 100, m * 100, "-", color=c, label=f"{lab}  load-aware (S3)")
    ax.fill_between(S_GRID * 100, a * 100, b * 100, color=c, alpha=.15)
    m2 = curves[(lab, "S2")][0]
    ax.plot(S_GRID * 100, m2 * 100, "--", color=c, lw=1, label=f"{lab}  link failover (S2)")
ax.axhline(TARGET * 100, color="k", lw=.8, ls=":")
ax.text(1, TARGET * 100 + .5, f"{TARGET:.0%} target", fontsize=8)
ax.set(xlabel="spare fallback capacity (% of high-band users' basic demand)",
       ylabel="users meeting minimum service (%)",
       title=f"Fallback capacity sweep (68-69 GHz -> 5 GHz, headroom {HEADROOM}x)\n"
             "S = 0 is no redistribution (S1); shading = 95% CI, bootstrap over rain days")
ax.legend(fontsize=6.8, ncol=2, loc="lower right"); ax.grid(alpha=.3)
ax.set_ylim(min(55, ax.get_ylim()[0]), 101)
fig.tight_layout(); fig.savefig(OUT / "fig2_spare_capacity_sweep.png", dpi=150)

# ------------------------------------------------------------------ Fig 3: recovery curves
EV0, EV1 = pd.Timestamp("2024-05-27 16:30"), pd.Timestamp("2024-05-27 21:30")
hc, fc = "cml22_sublink_1", "cml22_sublink_2"
gap = Lraw.isna().mean(axis=1) > 0.5
win = Lraw.loc[EV0:EV1, [hc, fc]]
assert not gap.loc[EV0:EV1].any(), "logging gap inside event window"
pre = Lraw.loc[EV0 - pd.Timedelta("72h"):EV0 - pd.Timedelta("6h"), [hc, fc]].median()
snr0 = {hc: pre[hc] - NOISE["68-69GHz"], fc: pre[fc] - NOISE["5GHz"]}


def g_of(c):
    A = (pre[c] - win[c]).clip(lower=0)
    g = se(snr0[c] - A) / se(snr0[c])
    return np.where(win[c].isna(), 0.0, g)


gH, gF = g_of(hc), g_of(fc)
T = len(win)
sl = {k: np.zeros(T) for k in ["S1", "S2", "S3"]}
onF2, healthy = False, 0
for t in range(T):
    capH = HEADROOM * gH[t]
    sl["S1"][t] = min(1, capH)
    # S2: decision from previous minute
    prev = HEADROOM * gH[t - 1] if t else HEADROOM
    if not onF2 and prev < 1:
        onF2, healthy = True, 0
    elif onF2:
        healthy = healthy + 1 if prev >= 1 else 0
        if prev >= 1 and healthy >= HOLD_MIN:
            onF2 = False
    sl["S2"][t] = min(1, S_EVENT * gF[t]) if onF2 else min(1, capH)
    # S3: assignment from previous minute, actual capacity this minute
    assignH = min(1, prev)
    sl["S3"][t] = min(assignH, capH) + min(1 - assignH, S_EVENT * gF[t])

met = []
for k, v in sl.items():
    met.append(dict(strategy=k, min_service=v.min(), user_minutes_lost_pct=(1 - v).sum(),
                    minutes_below_95=(v < .95).sum(), mean_service=v.mean()))
met = pd.DataFrame(met)
met.round(3).to_csv(OUT / "fig3_recovery_metrics.csv", index=False)
print(met.round(3).to_string())

# rain at hop 22 (3 nearest good PWS) from step-2 table
rain = LR[(LR.col == hc)].set_index(pd.to_datetime(LR[LR.col == hc].time_utc)).rain_mmh.loc[EV0:EV1]
fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True, gridspec_kw=dict(height_ratios=[1, 1, 2.2]))
ax[0].bar(rain.index, rain.values, width=0.003, color="tab:blue"); ax[0].set(ylabel="rain (mm/h)")
ax[0].set_title(f"Recovery curves, 2024-05-27 storm, hop 22 (measured RSL): spare S = {S_EVENT:.0%}, headroom {HEADROOM}x")
ax[1].plot(win.index, HEADROOM * gH, label="69 GHz capacity / demand")
ax[1].plot(win.index, S_EVENT * gF, label="5.7 GHz spare / demand")
ax[1].axhline(1, color="k", lw=.7, ls=":"); ax[1].set(ylabel="capacity"); ax[1].legend(fontsize=7)
names = {"S1": "S1 no redistribution", "S2": f"S2 link failover (hold {HOLD_MIN} min)", "S3": "S3 load-aware partial"}
for k, c in zip(["S1", "S2", "S3"], ["tab:grey", "tab:orange", "tab:green"]):
    r = met.set_index("strategy").loc[k]
    ax[2].plot(win.index, sl[k] * 100, color=c, lw=1.6 if k == "S3" else 1.2,
               label=f"{names[k]}: min {r.min_service:.0%}, {int(r.minutes_below_95)} min < 95%")
ax[2].axhline(95, color="k", lw=.7, ls=":")
ax[2].set(ylabel="users meeting minimum service (%)", xlabel="UTC"); ax[2].legend(fontsize=7.5, loc="lower right")
for x in ax: x.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "fig3_recovery_curves.png", dpi=150)

pd.Series(dict(HEADROOM=HEADROOM, S_EVENT=S_EVENT, HOLD_MIN=HOLD_MIN, TARGET=TARGET, N_BOOT=N_BOOT,
               SE_model="0.6*log2(1+SNR), max 4.4 b/s/Hz, 0 below -10 dB", noise_dBm=str(NOISE))).to_csv(OUT / "assumptions.csv")
print("done ->", OUT)
