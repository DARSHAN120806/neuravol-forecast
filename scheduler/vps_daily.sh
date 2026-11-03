#!/usr/bin/env bash
# NeuraVol daily update for a Linux VPS (called by cron or n8n).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
{
  echo "===== $(date) ====="
  .venv/bin/python daily_update.py
  [ -d data/live ] && .venv/bin/python collector/build_mid_surface.py
  git add data/surfaces data/forecasts data/latest_chain.csv
  [ -d data/surfaces_mid ] && git add data/surfaces_mid
  git diff --cached --quiet || git commit -m "daily update $(date +%F)"
  git push
} >> logs/daily_update.log 2>&1
