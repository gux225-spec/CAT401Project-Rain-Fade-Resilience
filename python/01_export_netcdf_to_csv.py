r"""
Step 1: export OpenMesh link RSL and full WU PWS NetCDF files to plain CSV.gz
so later analysis needs only pandas/numpy.

Needs numpy, pandas, netCDF4 (the only step that reads NetCDF).

Inputs : paths.LINKS_NC, paths.LINKS_META, paths.PWS_NC
Outputs: paths.PROCESSED
    links_meta.csv          103 real sublinks: id, freq, band, length, coordinates
    links_rsl_1min.csv.gz   wide table: time_utc + one RSL column (dBm) per sublink
    pws_stations.csv        37 stations: id, lat, lon, elevation
    pws_5min.csv.gz         long table: station, time_utc, rain rate/amount, temp, RH

Run:  python python/01_export_netcdf_to_csv.py
"""
import sys
import numpy as np
import pandas as pd
from paths import LINKS_NC, LINKS_META, PWS_NC, PROCESSED

try:
    from netCDF4 import Dataset, num2date
except ImportError:
    sys.exit("netCDF4 missing. In Anaconda Prompt run:  conda install -c conda-forge netcdf4")

OUT = PROCESSED


def arr(v):
    return np.asarray(np.ma.filled(v[:].astype(float), np.nan))


def to_time(v):
    units = getattr(v, "units", None)
    raw = np.asarray(v[:])
    if units:
        t = num2date(raw, units, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
        return pd.to_datetime(list(t), utc=True)
    return pd.to_datetime(raw, unit="s", utc=True)


def band(f_mhz):
    if f_mhz < 10000:
        return "5GHz"
    if f_mhz < 30000:
        return "24GHz"
    if f_mhz < 67000:
        return "60GHz"
    return "68-69GHz"


def export_links():
    meta = pd.read_csv(LINKS_META).drop(columns=["Unnamed: 0"], errors="ignore")
    meta["band"] = meta.frequency.map(band)
    meta["col"] = "cml" + meta.cml_id.astype(str) + "_" + meta.sublink_id
    with Dataset(LINKS_NC) as n:
        t = to_time(n["time"])
        cids = [str(x) for x in n["cml_id"][:]]
        subs = [str(x) for x in n["sublink_id"][:]]
        rsl = n["rsl"]
        cols = {}
        for _, r in meta.iterrows():
            i, j = cids.index(str(r.cml_id)), subs.index(r.sublink_id)
            cols[r.col] = np.round(arr(rsl[i, j, :]), 1)
    wide = pd.DataFrame(cols)
    wide.insert(0, "time_utc", t.strftime("%Y-%m-%d %H:%M"))
    wide.to_csv(OUT / "links_rsl_1min.csv.gz", index=False, compression="gzip")
    meta.to_csv(OUT / "links_meta.csv", index=False)
    print(f"links: {wide.shape}, {t.min()} -> {t.max()}")


def export_pws():
    frames, stations = [], []
    keep = ["rainfall_rate", "rainfall_amount", "temperature", "relative_humidity"]
    with Dataset(PWS_NC) as n:
        for sid, g in n.groups.items():
            df = pd.DataFrame({"station": sid, "time_utc": to_time(g["time"]).strftime("%Y-%m-%d %H:%M")})
            for k in keep:
                if k in g.variables:
                    df[k] = np.round(arr(g[k]).ravel()[: len(df)], 3)
            frames.append(df)
            info = {"station": sid}
            for a in ("latitude", "longitude", "altitude", "elevation"):
                if a in g.variables:
                    info[a] = float(np.ravel(g[a][:])[0])
                elif a in g.ncattrs():
                    info[a] = g.getncattr(a)
            stations.append(info)
    pws = pd.concat(frames, ignore_index=True)
    pws.to_csv(OUT / "pws_5min.csv.gz", index=False, compression="gzip")
    pd.DataFrame(stations).to_csv(OUT / "pws_stations.csv", index=False)
    print(f"pws: {pws.shape}, stations={len(stations)}, {pws.time_utc.min()} -> {pws.time_utc.max()}")


if __name__ == "__main__":
    export_links()
    export_pws()
    print("done ->", OUT)
