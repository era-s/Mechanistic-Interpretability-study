set -eu
JOB=/home/ll/codex-jobs/mechanistic-interpretability-study
cd "$JOB"
tar -xzf /mnt/c/Users/th312/mi-study-src.tar.gz
export PATH="$JOB/.venv/bin:$PATH"
export MPLBACKEND=Agg
python scripts/execute_notebook.py
