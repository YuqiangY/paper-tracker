#!/bin/bash
set -e

cd "$(dirname "$0")"

git add site/
if git diff --cached --quiet; then
    echo "No site changes to deploy."
    exit 0
fi

git commit -m "update: site $(date +%Y-%m-%d)"
git push origin main
echo "Site deployed successfully."
