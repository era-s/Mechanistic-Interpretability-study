set -eu
JOB=/home/ll/codex-jobs/mechanistic-interpretability-study
mkdir -p "$JOB"
python3 -m venv "$JOB/.venv"
"$JOB/.venv/bin/pip" install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1
"$JOB/.venv/bin/pip" install transformers==4.57.6 numpy==2.2.6 plotly==6.3.1 matplotlib==3.10.7 nbformat==5.10.4 nbclient==0.10.2 ipykernel==6.30.1 ipywidgets==8.1.7 jupyterlab==4.4.9
"$JOB/.venv/bin/python" -c 'import torch; print(torch.__version__, torch.cuda.get_device_name()); print((torch.ones(4, device="cuda") * 2).tolist())'
