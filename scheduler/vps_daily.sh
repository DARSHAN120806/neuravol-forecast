#!/usr/bin/env bash
# NeuraVol daily update for a Linux VPS (called by cron or n8n).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
{
  echo "===== $(date) ====="
  .venv/bin/python daily_update.py
  git add data/surfaces data/forecasts data/latest_chain.csv
  git diff --cached --quiet || git commit -m "daily update $(date +%F)"
  git push
} >> logs/daily_update.log 2>&1
