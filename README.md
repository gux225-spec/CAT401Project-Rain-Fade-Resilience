# Rain fade and recovery of multi-band wireless links

CAT401 / CEE 331 (Catastrophe Modeling and Resilience, Lehigh University) term project, Guohao Xiang.

A resilience assessment of dual-band wireless links (60–70 GHz primary + 5 GHz backup) under rain,
built on measured NYC Mesh link data (OpenMesh). It answers a planning question: **how much spare
backup capacity should be reserved, and which switching policy keeps the most users at minimum
service during and after a storm?**

> Quick guide : [GUIDE.md](GUIDE.md).
>
> This folder holds **code only**. No dataset is included (size limits and Weather Underground
> terms forbid redistribution). See [Data](#data) for what to download and where to put it.

## Repository layout

```
07_GitHub_Upload/
├── python/                 analysis pipeline (run in order 01 → 08)
│   ├── paths.py            every input/output location (absolute paths, edit once)
│   ├── common.py           shared loader for the step-2 binary table
│   ├── 01_export_netcdf_to_csv.py     NetCDF → CSV (needs netCDF4)
│   ├── 02_rain_attenuation_exploration.py   weather state, storm catalogue, attenuation, link pairs
│   ├── 03_measured_performance.py     throughput by band, normal vs rain (Fig. 1)
│   ├── 04_fallback_model.py           first fallback model, random backup pairing (Figs. 2–3)
│   ├── 05_validation_and_risk.py      fragility fit + held-out validation (Figs. 4–5)
│   ├── 06_bivariate_im_risk.py        peak + 24-h depth IM, hazard vs Atlas 14, annual risk (Figs. 8–9)
│   ├── 07_real_pair_fallback.py       strategies on real same-hop pairs, recovery, sensitivity (Figs. 10–12)
│   └── 08_physics_calibration.py      calibrated ITU physics, wet antenna, physics fragility (Fig. 13)
├── matlab/                 interactive ideal-physics simulator (base MATLAB R2020b+)
│   ├── RainFadeSimulator.m  GUI: custom rain rate / duration, ideal or calibrated parameters
│   ├── rainFadeModel.m      model core (no graphics)
│   ├── rainFadeDefaults.m   default scenario (OpenMesh hop 22)
│   ├── ituP838.m, specEff.m ITU-R P.838-3 coefficients, attenuated-Shannon capacity
│   └── rainFadeSelfTest.m   checks against ITU tables and the Python implementation
├── requirements.txt
└── .gitignore
```

## Data

All inputs are read from one folder, `06_Datasets_Raw_Adopted`, set in `python/paths.py`
(default `D:\CAT401\05_Term_Project\06_Datasets_Raw_Adopted`). Files are used exactly as downloaded.

| Folder | File(s) | Source |
|---|---|---|
| `OpenMesh_v1_links/` | `ds_openmesh.nc`, `links_metadata.csv`, `README.txt` | OpenMesh v1, Zenodo [10.5281/zenodo.15287692](https://doi.org/10.5281/zenodo.15287692) (CC BY 4.0) |
| `WU_PWS_NYC/` | `pws_wu_os.nc`, `pws_metadata.csv`, `README.md` | OpenMesh PWS, Zenodo [10.5281/zenodo.17508286](https://doi.org/10.5281/zenodo.17508286) (Weather Underground terms: personal, non-commercial, no redistribution) |
| `ASOS_1min/` | `asos_1min_precip_NYC_JFK_LGA_20231029_20240701.csv` | Iowa Environmental Mesonet ASOS 1-min, stations NYC/JFK/LGA, `vars=precip`, 2023-10-29 → 2024-07-02 UTC |
| `NOAA_Atlas14_NYC/` | intensity and depth PDS tables | NOAA Atlas 14 Vol. 10 v3, PFDS, lat 40.7789 lon −73.9692 (retrieved 2026-10-20) |
| `SIGMETRICS26_drive_tests/` | `alaska_road_test_202406/*_xcal_smart_tput.csv`, `hawaii_road_test_202408/*_xcal_smart_tput.csv`, `la_to_omaha_road_test_202411/xcal_smart_tput.tcp_downlink.*.csv`, `LICENSE-DATA`, `README.md` | [NUWiNS SIGMETRICS’26 release](https://github.com/NUWiNS/sigmetrics26-exploring-the-5g-digital-divide-in-the-non-contiguous-us-leo-satellites-to-the-rescue) |

ASOS download link used:
`https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py?station=NYC&station=JFK&station=LGA&vars=precip&sts=2023-10-29T00:00Z&ets=2024-07-02T00:00Z&sample=1min&what=download&delim=comma&gis=no`

## Setup and run

```bash
conda create -n cat401 python=3.11
conda activate cat401
pip install -r requirements.txt
```

Edit `PROJECT` in `python/paths.py` if the project is not at `D:\CAT401\05_Term_Project`
(or set the environment variable `CAT401_PROJECT`). Then run the steps in order:

```bash
python python/01_export_netcdf_to_csv.py
python python/02_rain_attenuation_exploration.py
python python/03_measured_performance.py
python python/04_fallback_model.py
python python/05_validation_and_risk.py
python python/06_bivariate_im_risk.py
python python/07_real_pair_fallback.py
python python/08_physics_calibration.py
```

Generated data go to `02_Data_Processed/`, tables and figures to `04_Results_Analysis/<step>/` (index: `04_Results_Analysis/README.md`).
Step 2 also writes a ~250 MB binary table (`link_rain_5min.pkl`); set `CAT401_CACHE` to put it on
another disk. Step 8 reads the fragility parameters written by step 5.

## MATLAB simulator

```matlab
cd matlab
rainFadeSelfTest      % should print "all checks passed"
RainFadeSimulator     % opens the app
```

Choose *Ideal physics* or the *OpenMesh calibrated* parameter sets (wet antenna 1.35 dB, path factor
1.06 Oct–Mar / 0.68 Apr–Jun, drying time constant 29 min) and change rain rate, duration, link and
network settings; the time response, rain-rate sweep and model notes update live.

## Main assumptions

- Capacity is estimated from received signal strength (SNR → attenuated-Shannon spectral efficiency),
  not measured user throughput. Dry-weather capacity = 1.5 × basic demand (headroom); varied 1.2–3×.
- A missing signal outside network-wide logging gaps is treated as a proxy outage (DS2).
- The record covers 29 Oct 2023 – 1 Jul 2024 (no July–September convective season); annual-risk
  results are exploratory.
