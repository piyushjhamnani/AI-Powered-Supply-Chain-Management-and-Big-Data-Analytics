"""Central settings for the Supply Chain Agent. Change numbers here, not inside agents."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Forecasting
FORECAST_HORIZON = 28          # days ahead (model uses lags >= 28 days, so it can forecast 28 days directly)
HOLDOUT_START = "2026-09-01"   # used to measure forecast accuracy

# Inventory policy
SERVICE_LEVEL_Z = 1.65         # 95% service level for safety stock
REVIEW_DAYS = {"fresh": 1, "staple": 7, "impulse": 7}
OVERSTOCK_DAYS = 30            # non-perishables above this cover are "overstock"
STOCKOUT_PENALTY_FACTOR = 1.5  # lost margin x this (customer goodwill)

# Purchase orders
AUTO_APPROVE_LIMIT = 50_000    # autopilot approves POs below this value (INR)

# Transfers
TRANSFER_DAYS = 1              # all warehouses are within ~250 km
DONOR_KEEP_DAYS = 20           # a donor warehouse keeps at least this many days of cover

# Anomaly detection
SPIKE_RATIO = 1.8
DROP_RATIO = 0.5

PERISHABLE_GROUP = {"P001": "fresh", "P002": "fresh", "P003": "fresh",
                    "P010": "impulse", "P012": "impulse", "P013": "impulse"}


def product_group(pid: str) -> str:
    return PERISHABLE_GROUP.get(pid, "staple")
