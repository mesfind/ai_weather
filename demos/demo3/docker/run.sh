#!/usr/bin/env bash
# Start Demo 3 on a Spark from the image built by docker/Dockerfile.
#   bash docker/run.sh          code baked into the image (what each country's Spark runs)
#   DEV=1 bash docker/run.sh    use the code in this folder instead, so edits show up live
# Then open http://localhost:8501 on the Spark (or tunnel: ssh -L 8501:localhost:8501 <spark>).
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUTS="${DEMO3_OUTPUTS:-$here/outputs}"         # saved forecasts, movies, job logs
CACHE="${DEMO3_CACHE:-$HOME/.cache/demo3}"         # model weights (~90 GB once all are fetched)
mkdir -p "$OUTPUTS" "$CACHE"

code=()
[ "${DEV:-0}" = 1 ] && code=(-v "$here:/app")

docker run -d --name demo3 --restart unless-stopped \
  --device nvidia.com/gpu=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 127.0.0.1:8501:8501 \
  "${code[@]}" -v "$OUTPUTS:/app/outputs" -v "$CACHE:/root/.cache" \
  -e DEMO3_IFS_SOURCE="${DEMO3_IFS_SOURCE:-aws}" \
  demo3
echo "Demo 3 starting: http://localhost:8501"
