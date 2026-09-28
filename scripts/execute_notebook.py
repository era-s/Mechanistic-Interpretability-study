"""Execute all notebook cells on the remote GPU and preserve outputs."""
from pathlib import Path
import json
import os
import datetime
import nbformat
from nbclient import NotebookClient

root=Path(__file__).resolve().parents[1]
path=root/"notebooks/01_residual_stream_qk_ov.ipynb"
job={"job_id":"mechanistic-interpretability-study", "host":"NG-GPU01 / Ubuntu-22.04",
     "remote_path":str(root),"pid":os.getpid(),"started_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
     "model":"openai-community/gpt2","status":"running","owns_server":False}
(root/"artifacts").mkdir(exist_ok=True)
job_path=root/"artifacts/job.json"
job_path.write_text(json.dumps(job,indent=2))
nb=nbformat.read(path,as_version=4)
def cell_finished(cell_index, **kwargs):
    job["last_completed_cell"]=cell_index
    job_path.write_text(json.dumps(job,indent=2))
    print("Completed cell",cell_index,flush=True)
try:
    NotebookClient(nb,timeout=180,kernel_name="python3",on_cell_executed=cell_finished,
                   resources={"metadata":{"path":str(root)}}).execute()
    nbformat.validate(nb)
    job["status"]="completed"
except Exception:
    job["status"]="failed"
    raise
finally:
    nbformat.write(nb,path)
    job["finished_at"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
    job_path.write_text(json.dumps(job,indent=2))
print("Executed and validated",path)
