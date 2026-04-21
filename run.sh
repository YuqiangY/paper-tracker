#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PATH="/home/yyq/miniforge3/bin:/home/yyq/.nvm/versions/node/v24.14.0/bin:$PATH"

source .env 2>/dev/null || true

DATE=$(date +%F)
mkdir -p data/daily

echo "=== Paper Tracker: $DATE ==="

python main.py 2>&1 | tee "data/daily/${DATE}.log"
PIPELINE_EXIT=${PIPESTATUS[0]}

if [ $PIPELINE_EXIT -ne 0 ]; then
    echo "ERROR: Pipeline failed with exit code $PIPELINE_EXIT"
    exit $PIPELINE_EXIT
fi

echo "=== Done ==="
