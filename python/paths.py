r"""
Absolute locations of every input and output used by the pipeline.

Edit PROJECT (or set the environment variable CAT401_PROJECT) if the project lives elsewhere.
Raw inputs are read ONLY from RAW (06_Datasets_Raw_Adopted, unchanged downloads); everything the
scripts create goes to PROCESSED and RESULTS.
"""
import os
from pathlib import Path

PROJECT = Path(os.environ.get("CAT401_PROJECT", r"D:\CAT401\05_Term_Project"))

# ---- raw datasets (read only) ---------------------------------------------------------------
RAW = PROJECT / "06_Datasets_Raw_Adopted"
LINKS_NC = RAW / "OpenMesh_v1_links" / "ds_openmesh.nc"            # OpenMesh v1, Zenodo 10.5281/zenodo.15287692
LINKS_META = RAW / "OpenMesh_v1_links" / "links_metadata.csv"
PWS_NC = RAW / "WU_PWS_NYC" / "pws_wu_os.nc"                       # OpenMesh PWS, Zenodo 10.5281/zenodo.17508286
PWS_META = RAW / "WU_PWS_NYC" / "pws_metadata.csv"
ASOS_CSV = RAW / "ASOS_1min" / "asos_1min_precip_NYC_JFK_LGA_20231029_20240701.csv"   # IEM ASOS 1-min
ATLAS_INTENSITY = RAW / "NOAA_Atlas14_NYC" / "noaa_atlas14_nyc_centralpark_pds_intensity_mm_per_h.csv"
ATLAS_DEPTH = RAW / "NOAA_Atlas14_NYC" / "noaa_atlas14_nyc_centralpark_pds_depth_mm.csv"
SIG_DIR = RAW / "SIGMETRICS26_drive_tests"                         # NUWiNS SIGMETRICS'26 release

# ---- generated data ------------------------------------------------------------------------
PROCESSED = PROJECT / "02_Data_Processed"                          # step 1-2 exports (CSV)
RESULTS = PROJECT / "04_Results_Analysis"                           # tables + figures, one subfolder per step
R2 = RESULTS / "02_rain_attenuation"
R5 = RESULTS / "05_validation_risk"
# large binary intermediate (~250 MB); set CAT401_CACHE to keep it on a fast local disk
CACHE = Path(os.environ.get("CAT401_CACHE", R2))


def result_dir(name):
    """Output folder of one step, created on first use."""
    d = RESULTS / name
    d.mkdir(parents=True, exist_ok=True)
    return d


for _d in (PROCESSED, R2, CACHE):
    _d.mkdir(parents=True, exist_ok=True)
