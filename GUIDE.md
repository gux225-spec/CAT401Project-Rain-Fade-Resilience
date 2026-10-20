# Model Operation Guide

## 1. Folder conventions

| Folder | Purpose | Modified by scripts? |
|---|---|---|
| `06_Datasets_Raw_Adopted` | Adopted raw datasets (as downloaded) | No, read-only |
| `02_Data_Processed` | Intermediate data exported by steps 1–2 | Yes |
| `04_Results_Analysis` | All result tables and figures, one subfolder per step | Yes |
| `07_GitHub_Upload` | Code (`python/`, `matlab/`), i.e. what goes to GitHub | No |

All paths are defined in `python/paths.py`. If the project moves, change only the `PROJECT` line there
(or set the environment variable `CAT401_PROJECT`).

## 2. Environment

```
conda create -n cat401 python=3.11
conda activate cat401
pip install -r requirements.txt
```

## 3. Run order

From the `07_GitHub_Upload` folder (Anaconda Prompt), run in order:

| Step | Command | What it does | Approx. time |
|---|---|---|---|
| 1 | `python python/01_export_netcdf_to_csv.py` | NetCDF → CSV (the only step that needs netCDF4) | 1–2 min |
| 2 | `python python/02_rain_attenuation_exploration.py` | Weather state, storm catalogue, attenuation, primary/backup pairs | 1–2 min |
| 3 | `python python/03_measured_performance.py` | Performance by band, normal vs rain | 1 min |
| 4 | `python python/04_fallback_model.py` | First fallback model (optional) | 30 s |
| 5 | `python python/05_validation_and_risk.py` | Fragility curves and validation | 3–5 min |
| 6 | `python python/06_bivariate_im_risk.py` | Two-parameter hazard, annual risk | 3–5 min |
| 7 | `python python/07_real_pair_fallback.py` | Strategies on real pairs, recovery, sensitivity | 1 min |
| 8 | `python python/08_physics_calibration.py` | Physics calibration (run step 5 first) | 1 min |

Step 2 writes a ~250 MB file `link_rain_5min.pkl` that steps 3–8 depend on. To keep it out of the
project, run `set CAT401_CACHE=E:\cache` (any folder) before running the scripts.

## 4. Parameters you are likely to change

| What | Where |
|---|---|
| Dry-weather headroom, spare capacity, detection delay, switch-back hold | `07_real_pair_fallback.py`, top: `BASE`, `S_GRID` |
| Spare-capacity levels, loss models | `06_bivariate_im_risk.py`, top: `S_LIST`, `MODELS` |
| Train/test split date | `SPLIT` in `05_validation_and_risk.py` and `08_physics_calibration.py` |
| Noise floor (signal strength → SNR) | `NOISE` at the top of each script |
| Rain-gauge quality-control thresholds | `qc["keep"]` line in `02_rain_attenuation_exploration.py` |

After a change, rerun from the modified step onward.

## 5. MATLAB simulator

```
cd('D:\CAT401\05_Term_Project\07_GitHub_Upload\matlab')
rainFadeSelfTest      % should print "all checks passed"
RainFadeSimulator     % opens the app
```

- **Preset** (left panel): pick NOAA Atlas 14 NYC 1/2/10/100-year storms, or drag rain rate and duration by hand.
- **Parameter set**: switch between ideal physics and OpenMesh-calibrated values
  (wet antenna 1.35 dB, path factor 1.06 / 0.68, drying time constant 29 min).
- Three tabs on the right: time response (rain → attenuation → capacity → service level),
  rain-rate sweep, and model notes.
- **Export CSV + PNG** saves the current scenario.

## 6. Common problems

- **`link_rain_5min.pkl` not found**: run step 2 first.
- **Step 8 cannot find `fragility_parameters.csv`**: run step 5 first.
- **`ModuleNotFoundError: netCDF4`**: `pip install netCDF4` (a conda install may hang while solving the environment).
- **MATLAB says a function is undefined**: the current folder is not `matlab/`, or MATLAB is older than R2020b.
