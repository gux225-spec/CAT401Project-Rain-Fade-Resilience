r"""
Step 2: rain events + link attenuation vs rain rate by band (OpenMesh, NYC).

Inputs : paths.PROCESSED (step 1), paths.ASOS_CSV, paths.PWS_META
Outputs: paths.RESULTS / 02_rain_attenuation  (link_rain_5min.pkl goes to paths.CACHE)
    weather_state_5min.csv.gz  dry / wet / unknown per 5-min window, with source
    paths.PROCESSED / link_pairs.csv   same-hop high/low-band pairs and their quality
    pws_qc.csv               PWS station QC (coverage, ratio to ASOS)
    rain_events.csv          citywide rain event catalog (ASOS JFK+LGA)
    link_rain_5min.pkl       long table: time, sublink, attenuation A_dB, missing fraction, rain
    atten_by_rain_bin.csv    median attenuation / missing-rate per band x rain bin
    fig_*.png

Run:  python python/02_rain_attenuation_exploration.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paths import PWS_META, ASOS_CSV, PROCESSED, result_dir
from common import LINK_RAIN

OUT = result_dir("02_rain_attenuation")
T0, T1 = "2023-10-29", "2024-07-01"

# ---------------------------------------------------------------- ITU-R P.838-3
_KH = dict(a=[-5.33980, -0.35351, -0.23789, -0.94158], b=[-0.10008, 1.26970, 0.86036, 0.64552],
           c=[1.13098, 0.45400, 0.15354, 0.16817], m=-0.18961, k=0.71147)
_KV = dict(a=[-3.80595, -3.44965, -0.39902, 0.50167], b=[0.56934, -0.22911, 0.73042, 1.07319],
           c=[0.81061, 0.51059, 0.11899, 0.27195], m=-0.16398, k=0.63297)
_AH = dict(a=[-0.14318, 0.29591, 0.32177, -5.37610, 16.1721], b=[1.82442, 0.77564, 0.63773, -0.96230, -3.29980],
           c=[-0.55187, 0.19822, 0.13164, 1.47828, 3.43990], m=0.67849, k=-1.95537)
_AV = dict(a=[-0.07771, 0.56727, -0.20238, -48.2991, 48.5833], b=[2.33840, 0.95545, 1.14520, 0.791669, 0.791459],
           c=[-0.76284, 0.54039, 0.26809, 0.116226, 0.116479], m=-0.053739, k=0.83433)


def _fit(p, f):
    x = np.log10(f)
    return sum(a * np.exp(-((x - b) / c) ** 2) for a, b, c in zip(p["a"], p["b"], p["c"])) + p["m"] * x + p["k"]


def itu_k_alpha(f_ghz, pol="v"):
    if pol == "h":
        return 10 ** _fit(_KH, f_ghz), _fit(_AH, f_ghz)
    return 10 ** _fit(_KV, f_ghz), _fit(_AV, f_ghz)


# self-check against P.838-3 Table 5 (horizontal): 20 GHz k=0.09164 a=1.0568; 60 GHz k=0.8606 a=0.7656
assert abs(itu_k_alpha(20, "h")[0] - 0.09164) < 2e-3 and abs(itu_k_alpha(60, "h")[1] - 0.7656) < 1e-2


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl, dp = np.radians(lon2 - lon1), p2 - p1
    return 6371e3 * 2 * np.arcsin(np.sqrt(np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2))


# ---------------------------------------------------------------- 1. ASOS
a = pd.read_csv(ASOS_CSV)
a["t"] = pd.to_datetime(a["valid(UTC)"])
a["mm"] = pd.to_numeric(a["precip"], errors="coerce") * 25.4
A1 = a.pivot_table(index="t", columns="station", values="mm").reindex(pd.date_range(T0, T1, freq="min"))
# only COMPLETE 5-min windows (5 valid minutes) give an accumulation; partial windows -> unknown
A5 = A1.resample("5min").sum(min_count=5)                      # mm per 5 min
asos_ref = A5[["JFK", "LGA"]].mean(axis=1)                      # mm per 5 min (NaN = unknown)

# ---------------------------------------------------------------- 2. PWS + QC
p = pd.read_csv(PROCESSED / "pws_5min.csv.gz", parse_dates=["time_utc"])
P = p.pivot_table(index="time_utc", columns="station", values="rainfall_amount")
P5 = P.resample("5min").sum(min_count=1).reindex(A5.index)
T5 = p.pivot_table(index="time_utc", columns="station", values="temperature").resample("5min").mean().reindex(A5.index)
mon = P5.resample("MS").sum(min_count=100)
ratio = mon.div(asos_ref.resample("MS").sum(), axis=0).median()
qc = pd.DataFrame({"coverage": P5.notna().mean(), "total_mm": P5.sum(), "ratio_to_asos": ratio})
qc["keep"] = qc.coverage.ge(0.5) & qc.ratio_to_asos.between(0.5, 2.0)
pm = pd.read_csv(PWS_META).rename(columns={"Station ID": "station"}).set_index("station")
qc = qc.join(pm[["Latitude", "Longitude"]])
pm.reset_index().to_csv(PROCESSED / "pws_stations.csv", index=False)     # authoritative coordinates
qc.round(3).to_csv(OUT / "pws_qc.csv")
good = qc.index[qc.keep]
print(f"PWS kept {len(good)}/{len(qc)}")

# ---------------------------------------------------------------- 3. rain events (ASOS)
# weather state per 5-min window: ASOS first; where ASOS is unknown use the QC'd PWS median
# (source kept); where both are unknown the state stays "unknown" (never filled as dry)
pws_med = P5[good].median(axis=1)
src = np.where(asos_ref.notna(), "ASOS", np.where(pws_med.notna(), "PWS", "none"))
city_mm = asos_ref.where(asos_ref.notna(), pws_med)              # mm per 5 min
state = pd.Series(np.where(city_mm.isna(), "unknown", np.where(city_mm > 0, "wet", "dry")), index=A5.index)
pd.DataFrame({"city_mm_5min": city_mm.round(3), "source": src, "state": state}).to_csv(
    OUT / "weather_state_5min.csv.gz", compression={"method": "gzip", "compresslevel": 1})
print("weather state:", state.value_counts().to_dict(), "| PWS-filled windows:", int((src == "PWS").sum()))
ref15 = city_mm.rolling(3, min_periods=3).sum() * 4               # 15-min rate, mm/h (complete only)
wet = state.eq("wet")
grp = (wet != wet.shift()).cumsum()
runs = pd.DataFrame({"wet": wet, "g": grp}).groupby("g").agg(wet=("wet", "first"), start=("wet", lambda s: s.index[0]),
                                                           end=("wet", lambda s: s.index[-1]), n=("wet", "size"))
# merge wet runs separated by < 3 h dry
events, cur = [], None
for _, r in runs.iterrows():
    if r.wet:
        if cur and (r.start - cur[1]) <= pd.Timedelta("3h"):
            cur[1] = r.end
        else:
            if cur: events.append(cur)
            cur = [r.start, r.end]
if cur: events.append(cur)
ev = []
for s, e in events:
    seg = city_mm[s:e]
    depth = seg.sum()
    if depth < 2.0:
        continue
    tc = T5[good].median(axis=1)[s:e].median()
    ev.append(dict(start=s, end=e + pd.Timedelta("5min"), hours=(e - s).total_seconds() / 3600 + 1 / 12,
                   depth_mm=depth, peak15_mmh=ref15[s:e].max(), median_temp_c=tc,
                   unknown_frac=float(state[s:e].eq("unknown").mean()),
                   pws_filled_frac=float((src[(A5.index >= s) & (A5.index <= e)] == "PWS").mean())))
ev = pd.DataFrame(ev)
ev["likely_snow_or_mixed"] = ev.median_temp_c < 2
ev.round(2).to_csv(OUT / "rain_events.csv", index=False)
print(f"events >=2 mm: {len(ev)}; peak15 max {ev.peak15_mmh.max():.1f} mm/h")

# ---------------------------------------------------------------- 4. links -> 5-min attenuation
meta = pd.read_csv(PROCESSED / "links_meta.csv")
L = pd.read_csv(PROCESSED / "links_rsl_1min.csv.gz", parse_dates=["time_utc"]).set_index("time_utc")
# network-wide logging gaps (>50% of all sublinks missing in the same minute) are a
# data-collection artefact, not a link outage -> excluded from all statistics
gap1 = L.isna().mean(axis=1) > 0.5
print(f"network-wide logging gaps: {gap1.mean():.1%} of minutes (excluded)")
L5 = L.resample("5min").mean().reindex(A5.index)
miss5 = L.isna().resample("5min").mean().reindex(A5.index)      # fraction of minutes missing
gap5 = gap1.resample("5min").mean().reindex(A5.index).fillna(1) > 0.5
L5[gap5] = np.nan
miss5[gap5] = np.nan

# citywide dry flag (no rain in ASOS ref or median PWS for the previous 2 h) -> baseline
city = pd.concat([asos_ref, pws_med], axis=1).max(axis=1)      # NaN only if every source is missing
dry = city.rolling(24, min_periods=24).sum().eq(0)                 # 2 h known-dry (unknown never counts as dry)
base = L5.where(dry).rolling("72h", min_periods=72).median().ffill()
Att = base - L5                                                  # dB, positive = extra loss

meta["mid_lat"] = (meta.site_0_lat + meta.site_1_lat) / 2
meta["mid_lon"] = (meta.site_0_lon + meta.site_1_lon) / 2
g = qc.loc[good]
rows = []
for _, m in meta.iterrows():
    d = haversine(m.mid_lat, m.mid_lon, g.Latitude.values, g.Longitude.values)
    order = np.argsort(d)[:3]                                    # 3 nearest good PWS
    rain = P5[g.index[order]].mean(axis=1, skipna=True)
    r15 = rain.rolling(3, min_periods=2).sum() * 4               # mm/h
    tmp = T5[g.index[order]].mean(axis=1)
    f = m.frequency / 1000
    k, al = itu_k_alpha(f, m.polarization)
    rows.append(pd.DataFrame({"time_utc": A5.index, "col": m.col, "band": m.band, "f_ghz": f, "len_km": m.length / 1000,
                              "A_dB": Att[m.col].values, "miss_frac": miss5[m.col].values,
                              "rain_mmh": r15.values, "temp_c": tmp.values,
                              "pws_dist_m": d[order[0]], "itu_k": k, "itu_a": al}))
LR = pd.concat(rows, ignore_index=True); del rows
for c_ in ["col", "band"]:
    LR[c_] = LR[c_].astype("category")
for c_ in ["f_ghz", "len_km", "A_dB", "miss_frac", "rain_mmh", "temp_c", "pws_dist_m", "itu_k", "itu_a"]:
    LR[c_] = LR[c_].astype("float32")
LR = LR[LR.miss_frac.notna()]                                    # drop logging-gap windows
# signal present but no dry baseline yet -> attenuation unknown (NOT zero); dropped and counted
nob = LR.A_dB.isna() & (LR.miss_frac <= 0.5)
qc_nob = pd.DataFrame({"band": LR.band, "no_baseline": nob}).groupby("band", observed=True).no_baseline.agg(["sum", "size"])
qc_nob.to_csv(OUT / "qc_no_baseline_windows.csv")
print(f"windows with signal but no baseline (dropped): {int(nob.sum())}")
LR = LR[~nob]
LR = LR[LR.temp_c.isna() | (LR.temp_c > 2)]                     # drop likely frozen precip
LR["A_itu_dB"] = LR.itu_k * LR.rain_mmh.clip(lower=0) ** LR.itu_a * LR.len_km
LR["A_itu_dB"] = LR.A_itu_dB.astype("float32")
LR.to_pickle(LINK_RAIN)                          # binary: fast to write/read (see common.py)

bins = [-0.01, 0, 2, 5, 10, 20, 40, 300]
labels = ["0 (dry)", "0-2", "2-5", "5-10", "10-20", "20-40", ">40"]
LR["rbin"] = pd.cut(LR.rain_mmh, bins, labels=labels)
summ = LR.groupby(["band", "rbin"], observed=True).agg(
    n=("A_dB", "size"), A_med=("A_dB", "median"), A_p90=("A_dB", lambda s: s.quantile(.9)),
    A_itu_med=("A_itu_dB", "median"), miss_rate=("miss_frac", "mean"),
    p_A_gt10=("A_dB", lambda s: (s > 10).mean())).reset_index()
summ.round(3).to_csv(OUT / "atten_by_rain_bin.csv", index=False)
print(summ.round(2).to_string())

# ---------------------------------------------------------------- 5. figures
bands = ["5GHz", "24GHz", "60GHz", "68-69GHz"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for b in bands:
    s = summ[summ.band == b]
    ax[0].plot(s.rbin.astype(str), s.A_med, "o-", label=f"{b} measured")
    ax[0].plot(s.rbin.astype(str), s.A_itu_med, "x--", alpha=.6, label=f"{b} ITU-R P.838")
    ax[1].plot(s.rbin.astype(str), s.miss_rate * 100, "o-", label=b)
ax[0].set(xlabel="15-min rain rate at link (mm/h)", ylabel="median extra path loss (dB)", title="Attenuation vs rain")
ax[1].set(xlabel="15-min rain rate at link (mm/h)", ylabel="% of minutes with no RSL", title="Signal-missing rate vs rain")
ax[0].legend(fontsize=7); ax[1].legend(fontsize=8)
for x in ax: x.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "fig_atten_missing_vs_rain.png", dpi=150)

# example event on dual-band hop 22 (5.7 GHz + 69 GHz), heaviest warm event
e = ev[~ev.likely_snow_or_mixed].sort_values("peak15_mmh").iloc[-1]
w0, w1 = e.start - pd.Timedelta("6h"), e.end + pd.Timedelta("6h")
fig, ax = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
for c, lab in [("cml22_sublink_1", "69.1 GHz"), ("cml22_sublink_2", "5.7 GHz")]:
    ax[0].plot(Att.loc[w0:w1, c], label=lab)
sub = LR[(LR.col == "cml22_sublink_1")].set_index("time_utc").loc[w0:w1]
ax[0].plot(sub.A_itu_dB, "k--", lw=1, label="ITU-R P.838 (69 GHz)")
ax[0].set(ylabel="extra loss (dB)", title=f"Hop 22, event {e.start:%Y-%m-%d} (peak {e.peak15_mmh:.0f} mm/h)")
ax[0].legend(fontsize=8)
ax[1].bar(sub.index, sub.rain_mmh, width=0.003, color="tab:blue")
ax[1].set(ylabel="rain (mm/h)")
for x in ax: x.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "fig_event_hop22.png", dpi=150)

fig, ax = plt.subplots(figsize=(10, 3))
ax.bar(ev.start, ev.peak15_mmh, width=1.0, color=np.where(ev.likely_snow_or_mixed, "grey", "tab:blue"))
ax.set(ylabel="peak 15-min rate (mm/h)", title=f"{len(ev)} precipitation events >= 2 mm (ASOS JFK+LGA; grey = likely snow/mixed)")
ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(OUT / "fig_event_catalog.png", dpi=150)
print("done ->", OUT)

# ---------------------------------------------------------------- 6. real primary/backup pairs (same hop, same direction)
cov = L.notna().mean()
dry1 = dry.reindex(L.index, method="ffill").fillna(False).astype(bool).values
dry_sd = L.loc[dry1].std()
pairs = []
for cid, g_ in meta.groupby("cml_id"):
    hi = g_[g_.band.isin(["60GHz", "68-69GHz"])]
    lo = g_[g_.band == "5GHz"]
    for _, h_ in hi.iterrows():
        for _, l_ in lo.iterrows():
            dsame = haversine(h_.site_0_lat, h_.site_0_lon, l_.site_0_lat, l_.site_0_lon)
            dopp = haversine(h_.site_0_lat, h_.site_0_lon, l_.site_1_lat, l_.site_1_lon)
            pairs.append(dict(pair_id=f"cml{cid}:{h_.sublink_id}->{l_.sublink_id}", cml_id=cid, high=h_.col, low=l_.col,
                              f_high=h_.frequency, f_low=l_.frequency, length_m=h_.length,
                              direction="same" if dsame < 100 else ("opposite" if dopp < 100 else "unclear"),
                              cov_high=cov[h_.col], cov_low=cov[l_.col], dry_sd_high=dry_sd[h_.col], dry_sd_low=dry_sd[l_.col]))
pairs = pd.DataFrame(pairs)
pairs["usable"] = (pairs.direction == "same") & (pairs.cov_low >= 0.8) & (pairs.dry_sd_low <= 3) & (pairs.cov_high >= 0.8)
pairs.round(3).to_csv(PROCESSED / "link_pairs.csv", index=False)
print(pairs.round(2).to_string())
