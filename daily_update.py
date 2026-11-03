"""Daily pipeline: bring surfaces and forecast up to date (used by the evening scheduler).

    python daily_update.py                 # sync every missing session up to the latest completed one
    python daily_update.py --no-forecast
    python daily_update.py --source nse    # auto (default) | nse | upstox

Same engine as the admin panel's "Sync now" button (src/sync.py): NSE bhavcopy for each missing day, Upstox
live quotes as a fallback for the latest session, then a fresh forecast.
"""
import argparse
from datetime import date

from src import sync


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-forecast", action="store_true")
    ap.add_argument("--until")
    ap.add_argument("--source", default="auto", choices=["auto", "nse", "upstox"])
    a = ap.parse_args()
    res = sync.run_sync(until=date.fromisoformat(a.until) if a.until else None, source=a.source,
                        update_forecast=not a.no_forecast)
    print(res.message)


if __name__ == "__main__":
    main()
