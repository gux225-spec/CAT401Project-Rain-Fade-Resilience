"""Shared loader for the step-2 site/window table (fast binary format)."""
import pandas as pd
from paths import CACHE

CACHE.mkdir(parents=True, exist_ok=True)
LINK_RAIN = CACHE / "link_rain_5min.pkl"


def load_link_rain(cols=None, bands=None):
    """time_utc (datetime64), col/band (category), float32 values. Windows with signal but no
    dry baseline are already removed in step 2; logging-gap windows are removed too."""
    df = pd.read_pickle(LINK_RAIN)
    if bands is not None:
        df = df[df.band.isin(bands)]
    if cols is not None:
        df = df[cols]
    return df.reset_index(drop=True)
