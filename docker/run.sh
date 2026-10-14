#!/usr/bin/env bash
# docker build -t ai_weather -f docker/Dockerfile .
# Start forecasting and benchmarking Demos on a Spark from the image built by docker/Dockerfile.
#   bash docker/run.sh          code baked into the image (what each country's Spark runs)
#   DEV=1 bash docker/run.sh    use the code in this folder instead, so edits show up live
# Then open http://localhost:8501 on the Spark (or tunnel: ssh -L 8501:localhost:8501 <spark>).
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUTS="${DEMO_OUTPUTS:-$here/outputs}"         # saved forecasts, movies, job logs
CACHE="${DEMO_CACHE:-$HOME/.cache/ai_weather}"         # model weights (~90 GB once all are fetched)
mkdir -p "$OUTPUTS" "$CACHE"

code=()
[ "${DEV:-0}" = 1 ] && code=(-v "$here:/app")

docker run -d --name ai_weather --restart unless-stopped \
  --device nvidia.com/gpu=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 127.0.0.1:8501:8501 \
  "${code[@]}" -v "$OUTPUTS:/app/outputs" -v "$CACHE:/root/.cache" \
  -e DEMO_IFS_SOURCE="${DEMO3_IFS_SOURCE:-aws}" \
  ai_weather
echo "AI Forecasting starting: http://localhost:8501"
