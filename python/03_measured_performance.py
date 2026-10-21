r"""
Step 3 / Figure 1: measured per-band performance under normal vs rain labels.

(a) SIGMETRICS'26 XCAL downlink throughput (Alaska + Hawaii, the only files with a
    per-sample weather label) by serving technology x weather label.
    normal = sunny + cloudy, rain = rainy. Padded rows removed, duplicates on
    (run_id, utc_dt, direction, protocol) removed. NO SERVICE kept as a share.
    Mainland mmWave (LA->Omaha) has no per-sample weather label -> shown as reference only.
(b) OpenMesh: estimated relative capacity per band, dry vs rain (>= 2 mm/h), derived from
    measured RSL with an attenuated-Shannon mapping (NOT measured throughput).

Sufficiency rule: a cell is "insufficient" if it has < 300 samples or < 5 distinct runs.
Outputs -> paths.RESULTS / 03_measured_performance

Run:  python python/03_measured_performance.py
"""
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paths import SIG_DIR, PROCESSED, result_dir
from common import load_link_rain

OUT = result_dir("03_measured_performance")
MIN_N, MIN_RUNS = 300, 5
TECHS = ["LTE", "5G-low", "5G-mid", "5G-mmWave"]

# ------------------------------------------------------------- (a) SIGMETRICS
frames = []
for f in sorted(glob.glob(str(SIG_DIR / "*" / "*_xcal_smart_tput.csv"))):
    region = Path(f).parent.name.split("_")[0]
    op = Path(f).name.split("_")[0]
    d = pd.read_csv(f, low_memory=False,
                    usecols=["run_id", "utc_dt", "tput_dl", "actual_tech", "app_tput_direction",
                             "app_tput_protocol", "weather", "is_padded"])
    d = d[(d.app_tput_direction == "downlink") & (~d.is_padded.astype(bool))]
    d = d.drop_duplicates(["run_id", "utc_dt", "app_tput_direction", "app_tput_protocol"])
    d["region"], d["operator"] = region, op
    frames.append(d)
S = pd.concat(frames, ignore_index=True)
S["cond"] = np.where(S.weather.eq("rainy"), "rain", "normal")
S["tech"] = S.actual_tech.str.replace(r"5G-mmWave.*", "5G-mmWave", regex=True)

# mainland mmWave reference (no weather label)
mm = []
for f in glob.glob(str(SIG_DIR / "la_to_omaha_road_test_202411" / "xcal_smart_tput.tcp_downlink.*.csv")):
    d = pd.read_csv(f, low_memory=False, usecols=["run_id", "utc_ts", "actual_tech", "dl_tput_mbps"])
    mm.append(d[d.actual_tech.str.startswith("5G-mmWave", na=False)])
MM = pd.concat(mm).drop_duplicates(["run_id", "utc_ts"])

rows = []
for (t, c), g in S.groupby(["tech", "cond"]):
    rows.append(dict(tech=t, cond=c, n=len(g), runs=g.run_id.nunique(),
                     regions="+".join(sorted(g.region.unique())),
                     median=g.tput_dl.median(), p10=g.tput_dl.quantile(.1), p90=g.tput_dl.quantile(.9)))
for t in TECHS:
    for c in ["normal", "rain"]:
        if not any((r["tech"] == t) and (r["cond"] == c) for r in rows):
            rows.append(dict(tech=t, cond=c, n=0, runs=0, regions="", median=np.nan, p10=np.nan, p90=np.nan))
tab = pd.DataFrame(rows)
tab["sufficient"] = (tab.n >= MIN_N) & (tab.runs >= MIN_RUNS)
ns = S.assign(nos=S.tech.eq("NO SERVICE")).groupby("cond").nos.mean()
tab.round(2).sort_values(["tech", "cond"]).to_csv(OUT / "throughput_by_tech_weather.csv", index=False)
print(tab.round(1).sort_values(["tech", "cond"]).to_string())
print("NO SERVICE share:", ns.round(3).to_dict(), "| mainland mmWave ref n =", len(MM))

# ------------------------------------------------------------- (b) OpenMesh capacity proxy
LR = load_link_rain(["col", "band", "A_dB", "miss_frac", "rain_mmh"])
LR["col"] = LR.col.astype(str); LR["band"] = LR.band.astype(str)
meta = pd.read_csv(PROCESSED / "links_meta.csv")
NOISE = {"5GHz": -88.0, "24GHz": -84.0, "60GHz": -74.0, "68-69GHz": -74.0}  # dBm, see model notes


def se(snr_db):  # attenuated Shannon, 3GPP TR 36.942: alpha 0.6, SEmax 4.4 b/s/Hz, floor -10 dB
    s = 0.6 * np.log2(1 + 10 ** (np.asarray(snr_db) / 10))
    return np.where(np.asarray(snr_db) < -10, 0.0, np.minimum(s, 4.4))


med = pd.read_csv(PROCESSED / "links_rsl_1min.csv.gz", usecols=list(meta.col)).median()
meta["snr_dry"] = meta.col.map(med) - meta.band.map(NOISE)
LR = LR.merge(meta[["col", "snr_dry"]], on="col")
LR = LR.dropna(subset=["rain_mmh"])
LR["cap_rel"] = np.where(LR.miss_frac > 0.5, 0.0, se(LR.snr_dry - LR.A_dB.fillna(0).clip(lower=0)) / se(LR.snr_dry))
LR["cond"] = np.select([LR.rain_mmh == 0, LR.rain_mmh >= 2], ["dry", "rain >= 2 mm/h"], "light")
cap = LR[LR.cond != "light"].groupby(["band", "cond"]).cap_rel.describe(percentiles=[.05, .5]).round(3)
cap.to_csv(OUT / "openmesh_capacity_proxy.csv")
print(cap)

# ------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=[1.35, 1]))
pos, data, cols, labels = [], [], [], []
for i, t in enumerate(TECHS):
    for j, c in enumerate(["normal", "rain"]):
        x = i * 3 + j
        g = S[(S.tech == t) & (S.cond == c)].tput_dl
        r = tab[(tab.tech == t) & (tab.cond == c)].iloc[0]
        if len(g):
            b = ax[0].boxplot([g.clip(lower=0.1)], positions=[x], widths=0.7, patch_artist=True,
                              showfliers=False, medianprops=dict(color="k"))
            b["boxes"][0].set_facecolor("tab:blue" if c == "normal" else "tab:orange")
            b["boxes"][0].set_alpha(1 if r.sufficient else 0.3)
        if t == "5G-mmWave":
            continue
        txt = f"n={r.n}\nruns={r.runs}" + ("" if r.sufficient else "\nINSUFFICIENT")
        ax[0].text(x, 1500, txt, ha="center", va="top", fontsize=6.5, color="k" if r.sufficient else "red")
if len(MM):
    b = ax[0].boxplot([MM.dl_tput_mbps.clip(lower=0.1)], positions=[9.5], widths=0.8, patch_artist=True,
                      showfliers=False, medianprops=dict(color="k"))
    b["boxes"][0].set_facecolor("lightgrey")
    ax[0].text(9.5, 0.6, f"0 weather-labelled\nsamples (INSUFFICIENT)\ngrey = mainland ref,\nno weather label, n={len(MM)}",
               ha="center", va="bottom", fontsize=6.5, color="red")
ax[0].text(0.99, 0.02, "rain samples are mostly Hawaii; route/terrain/coverage confounded\n-> descriptive only, not a causal rain effect",
           transform=ax[0].transAxes, ha="right", fontsize=6.5, style="italic")
ax[0].set_xticks([i * 3 + .5 for i in range(4)], TECHS)
ax[0].set_yscale("log"); ax[0].set_ylim(0.1, 3000)
ax[0].set(ylabel="downlink throughput (Mbps, log)", title="(a) Measured DL throughput, SIGMETRICS'26 (AK+HI)")
ax[0].plot([], [], "s", color="tab:blue", label="normal (sunny/cloudy)")
ax[0].plot([], [], "s", color="tab:orange", label="rain label")
ax[0].legend(fontsize=7, loc="lower left")
ax[1].plot([], [], "s", color="tab:blue", label="dry"); ax[1].plot([], [], "s", color="tab:orange", label="rain >= 2 mm/h")
ax[1].legend(fontsize=7, loc="lower left"); ax[0].grid(alpha=.3, axis="y")

bands = ["5GHz", "24GHz", "60GHz", "68-69GHz"]
for i, bnd in enumerate(bands):
    for j, c in enumerate(["dry", "rain >= 2 mm/h"]):
        g = LR[(LR.band == bnd) & (LR.cond == c)].cap_rel
        b = ax[1].boxplot([g], positions=[i * 3 + j], widths=0.7, patch_artist=True, whis=(5, 95),
                          showfliers=False, medianprops=dict(color="k"))
        b["boxes"][0].set_facecolor("tab:blue" if j == 0 else "tab:orange")
        out = (g == 0).mean() * 100
        ax[1].text(i * 3 + j, 1.08, f"outage\n{out:.0f}%", ha="center", fontsize=6.5)
ax[1].set_xticks([i * 3 + .5 for i in range(4)], bands)
ax[1].set_ylim(-0.02, 1.2)
ax[1].set(ylabel="capacity relative to dry median", title="(b) OpenMesh: capacity estimated from RSL")
ax[1].grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig(OUT / "fig1_measured_performance.png", dpi=150)
print("done ->", OUT)
