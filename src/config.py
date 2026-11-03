"""Project-wide settings for NeuraVol Forecast."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
SURFACES = DATA / "surfaces"
REPORTS = ROOT / "reports"

SYMBOL = "NIFTY"

# ---- surface grid -------------------------------------------------------
TENORS_DAYS = [7, 14, 30, 60, 91, 182, 365]                       # rows
MONEYNESS = [-0.15, -0.10, -0.05, -0.025, 0.0, 0.025, 0.05, 0.10, 0.15]  # ln(K/F), columns

# ---- cleaning -----------------------------------------------------------
MIN_DAYS_TO_EXPIRY = 5      # drop very short expiries (noisy, pin risk)
MAX_DAYS_TO_EXPIRY = 400
MIN_PRICE = 0.5             # rupees; below this, tick size dominates the price
MIN_POINTS_PER_SLICE = 6    # need this many clean quotes to fit an SVI smile
MAX_ABS_MONEYNESS = 0.35
IV_BOUNDS = (0.02, 2.0)
MAX_REL_SVI_RMSE = 0.15     # drop a smile if fit error > 15% of its ATM vol
EXTRAP_MARGIN = 0.01        # trust SVI this far beyond the last quote
MAX_WING_SLOPE = 1.0        # cap on d(iv^2)/dk beyond the quotes (total-variance slope = this * T)

# ---- risk-free rate -----------------------------------------------------
# Approximation: RBI repo rate by effective date (proxy for the 3M T-bill).
# IV is insensitive to small rate errors (we price on the forward), but replace
# this with RBI 91-day T-bill yields (data/rates.csv: date,rate) when available.
REPO_RATE_HISTORY = [
    ("2015-09-29", 0.0675), ("2016-04-05", 0.0650), ("2016-10-04", 0.0625),
    ("2017-08-02", 0.0600), ("2018-06-06", 0.0625), ("2018-08-01", 0.0650),
    ("2019-02-07", 0.0625), ("2019-04-04", 0.0600), ("2019-06-06", 0.0575),
    ("2019-08-07", 0.0540), ("2019-10-04", 0.0515), ("2020-03-27", 0.0440),
    ("2020-05-22", 0.0400), ("2022-05-04", 0.0440), ("2022-06-08", 0.0490),
    ("2022-08-05", 0.0540), ("2022-09-30", 0.0590), ("2022-12-07", 0.0625),
    ("2023-02-08", 0.0650), ("2025-02-07", 0.0625), ("2025-04-09", 0.0600),
    ("2025-06-06", 0.0550),
]
