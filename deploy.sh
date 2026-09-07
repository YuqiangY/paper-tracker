#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

BRANCH="main"
MAX_RETRY=3

git add site/
if git diff --cached --quiet; then
    echo "No site changes to deploy."
    exit 0
fi

git commit -m "update: site $(date +%Y-%m-%d)"

# Push with retry — tolerate transient network/SSH failures
push_ok=0
for attempt in $(seq 1 "$MAX_RETRY"); do
    if git push origin "$BRANCH"; then
        push_ok=1
        break
    fi
    echo "WARN: push attempt $attempt/$MAX_RETRY failed, retrying in 5s..." >&2
    sleep 5
done

if [ "$push_ok" -ne 1 ]; then
    echo "ERROR: git push failed after $MAX_RETRY attempts. Commit is LOCAL-ONLY; site NOT deployed." >&2
    exit 1
fi

# Verify the remote actually received our commit — a 0 exit code is not proof
git fetch origin "$BRANCH" --quiet
local_head=$(git rev-parse HEAD)
remote_head=$(git rev-parse "origin/$BRANCH")
if [ "$local_head" != "$remote_head" ]; then
    echo "ERROR: push reported success but origin/$BRANCH ($remote_head) != local HEAD ($local_head)." >&2
    echo "       Site NOT deployed. Run 'git push origin $BRANCH' manually." >&2
    exit 1
fi

echo "Site deployed successfully. origin/$BRANCH is now at ${local_head:0:7}"
