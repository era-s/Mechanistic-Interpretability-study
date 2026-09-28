"""Small GPU checks beyond the notebook's main prompt."""
from pathlib import Path
import sys
import json
import numpy as np

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
import study

model,tokenizer=study.load_model()
results=[]
for prompt in ["Hello", "The capital of France is", "John John John"]:
    runs,error=study.experiment(model,tokenizer,prompt,new_tokens=1)
    r=runs[0]
    n=len(r["ids"])
    assert r["residual"].shape==(25,n,768)
    assert np.isfinite(r["residual"]).all()
    np.testing.assert_allclose(r["attention"].sum(-1),1.,atol=1e-6)
    assert np.max(np.triu(r["attention"],k=1))==0
    study.source_contributions(r,11,11,n-1)
    study.logit_lens(model,r,tokenizer)
    results.append(dict(prompt=prompt,tokens=n,prefix_error=error,checks=r["errors"]))
try:
    study.experiment(model,tokenizer,"",new_tokens=1)
    raise AssertionError("Empty prompt accepted")
except ValueError:
    pass
path=root/"artifacts/edge-case-validation.json"
path.write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
