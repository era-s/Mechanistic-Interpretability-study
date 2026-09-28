set -eu
JOB=/home/ll/codex-jobs/mechanistic-interpretability-study
cp /mnt/c/Users/th312/mi-check-edge-cases.py "$JOB/scripts/check_edge_cases.py"
cd "$JOB"
export PATH="$JOB/.venv/bin:$PATH"
python scripts/check_edge_cases.py
