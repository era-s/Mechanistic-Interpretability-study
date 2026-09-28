"""GPT-2 activation capture and study figures. All tensors use row vectors."""
from pathlib import Path
import json
import platform
import importlib.metadata

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import AutoTokenizer, GPT2LMHeadModel

MODEL_ID = "openai-community/gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


def load_model(device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment; connect to the GPU kernel.")
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = GPT2LMHeadModel.from_pretrained(
        MODEL_ID, revision=REVISION, attn_implementation="eager",
        torch_dtype=torch.float32,
    ).to(device).eval()
    assert (model.config.n_embd, model.config.n_layer, model.config.n_head) == (768, 12, 12)
    return model, tokenizer


def cpu(x):
    return x.detach().float().cpu().numpy()


@torch.inference_mode()
def capture(model, ids):
    """Capture one unpadded sequence; reconstruct all heads including affine biases."""
    if ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= ids.shape[1] <= 128:
        raise ValueError("Use one unpadded sequence of 1..128 tokens.")
    blocks = [{} for _ in model.transformer.h]
    handles = []
    for block, cache in zip(model.transformer.h, blocks):
        def before(module, args, c=cache):
            c["pre"] = args[0].detach()
        def ln1(module, args, output, c=cache):
            c["ln1"] = output.detach()
        def attn(module, args, output, c=cache):
            c["attn"] = output[0].detach()
        def mlp(module, args, output, c=cache):
            c["mlp"] = output.detach()
        def after(module, args, output, c=cache):
            c["post"] = output[0].detach()
        handles.extend([
            block.register_forward_pre_hook(before),
            block.ln_1.register_forward_hook(ln1),
            block.attn.register_forward_hook(attn),
            block.mlp.register_forward_hook(mlp),
            block.register_forward_hook(after),
        ])
    try:
        output = model(ids, use_cache=False, output_attentions=True)
    finally:
        for handle in handles:
            handle.remove()
    n = ids.shape[1]
    mask = torch.ones(n, n, device=ids.device, dtype=torch.bool).tril()
    stages, names = [cpu(blocks[0]["pre"][0])], ["embedding + position"]
    scores, patterns, writes, values, deltas = [], [], [], [], []
    errors = {"attention_pattern": 0., "attention_output": 0., "residual_sum": 0.}
    for layer, (block, c) in enumerate(zip(model.transformer.h, blocks)):
        q, k, v = block.attn.c_attn(c["ln1"]).split(768, dim=-1)
        q, k, v = [x[0].reshape(n, 12, 64).permute(1, 0, 2) for x in (q, k, v)]
        score = q @ k.transpose(-1, -2) / 8.0
        a = score.masked_fill(~mask, float("-inf")).softmax(-1)
        # HF Conv1D stores W[in, out]; each head owns 64 rows of W_O.
        wo = block.attn.c_proj.weight.reshape(12, 64, 768)
        source_ov = v @ wo
        head_write = a @ source_ov
        reconstructed = head_write.sum(0) + block.attn.c_proj.bias
        expected = c["pre"] + c["attn"] + c["mlp"]
        errors["attention_pattern"] = max(errors["attention_pattern"], (a-output.attentions[layer][0]).abs().max().item())
        errors["attention_output"] = max(errors["attention_output"], (reconstructed-c["attn"][0]).abs().max().item())
        errors["residual_sum"] = max(errors["residual_sum"], (expected-c["post"]).abs().max().item())
        torch.testing.assert_close(a, output.attentions[layer][0], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(reconstructed, c["attn"][0], atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(expected, c["post"], atol=2e-5, rtol=2e-5)
        stages.extend([cpu((c["pre"]+c["attn"])[0]), cpu(c["post"][0])])
        names.extend([f"L{layer:02d} + attention", f"L{layer:02d} + MLP"])
        scores.append(cpu(score.masked_fill(~mask, float("nan"))))
        patterns.append(cpu(a))
        writes.append(cpu(head_write))
        values.append(cpu(source_ov))
        deltas.append(np.stack([cpu(c["attn"][0]), cpu(c["mlp"][0])]))
    return dict(ids=cpu(ids[0]).astype(int), residual=np.stack(stages), stages=names,
                scores=np.stack(scores), attention=np.stack(patterns),
                head_writes=np.stack(writes), source_ov=np.stack(values),
                updates=np.stack(deltas), logits=cpu(output.logits[0]), errors=errors)


def labels(tokenizer, ids):
    return [f"{i}: {tokenizer.decode([int(t)])!r}" for i, t in enumerate(ids)]


@torch.inference_mode()
def experiment(model, tokenizer, prompt, new_tokens=6):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    if not 1 <= ids.shape[1] <= 128-new_tokens or not 0 <= new_tokens <= 16:
        raise ValueError("Prompt must be nonempty; total <=128 tokens, generation <=16.")
    runs = []
    for step in range(new_tokens+1):
        run = capture(model, ids)
        run["labels"] = labels(tokenizer, run["ids"])
        run["text"] = tokenizer.decode(ids[0])
        runs.append(run)
        if step < new_tokens:
            next_id = int(run["logits"][-1].argmax())
            ids = torch.cat([ids, ids.new_tensor([[next_id]])], dim=1)
    # A causal model cannot change previous token states when a suffix is appended.
    prefix_error = max(float(np.max(np.abs(r["residual"][:, :runs[0]["ids"].size] - runs[0]["residual"]))) for r in runs)
    assert prefix_error < 1e-3, prefix_error
    return runs, prefix_error


def animate(fig, frames, title, slider_label):
    fig.frames = frames
    fig.update_layout(title=title, template="plotly_white", height=600,
        margin=dict(l=110, r=80, t=90, b=140),
        updatemenus=[dict(type="buttons", direction="right", x=0, y=-.29, buttons=[
            dict(label="Play", method="animate", execute=False, args=[None, {"frame": {"duration": 650, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]),
            dict(label="Pause", method="animate", execute=False, args=[[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}])])],
        sliders=[dict(currentvalue={"prefix": slider_label+": "}, y=-.08,
            steps=[dict(label=f.name.replace("embedding + position","E").replace(" + attention","A").replace(" + MLP","M"), method="animate", execute=False, args=[[f.name], {"mode":"immediate", "frame":{"duration":0,"redraw":True}, "transition":{"duration":0}}]) for f in frames])])
    return fig


def residual_movie(run, scale="signed_log"):
    raw = run["residual"]
    if scale not in ("signed_log", "raw"):
        raise ValueError("scale must be signed_log or raw")
    r = np.sign(raw)*np.log1p(np.abs(raw)) if scale == "signed_log" else raw
    bound = float(np.max(np.abs(r)))
    def heat(z, original):
        return go.Heatmap(z=z, x=np.arange(768), y=run["labels"], zmin=-bound, zmax=bound,
            customdata=original,hovertemplate="%{y}<br>coordinate %{x}<br>activation %{customdata:.5f}<extra></extra>",
            colorscale="RdBu", reversescale=True, colorbar=dict(title="signed log1p" if scale=="signed_log" else "activation"))
    frames = [go.Frame(name=name, data=[heat(z,original)]) for name,z,original in zip(run["stages"], r,raw)]
    fig = animate(go.Figure(data=[heat(r[0],raw[0])]), frames, f"Residual stream: 768 coordinates / {scale} color", "depth")
    fig.update_xaxes(title="residual coordinate (not a named feature)")
    fig.update_yaxes(autorange="reversed", title="token position")
    return fig


def fit_projection(runs):
    # One basis across all stages AND generation steps; no per-frame refitting.
    x = np.concatenate([r["residual"].reshape(-1,768) for r in runs])
    mean = x.mean(0)
    _, singular, vt = np.linalg.svd(x-mean, full_matrices=False)
    return mean, vt[:3].T, (singular[:3]**2 / (singular**2).sum())


def trajectory(run, mean, basis, variance):
    projected = (run["residual"]-mean) @ basis
    fig = go.Figure()
    for i, label in enumerate(run["labels"]):
        p = projected[:,i]
        fig.add_trace(go.Scatter3d(x=p[:,0], y=p[:,1], z=p[:,2], mode="lines+markers",
            name=label, text=run["stages"], hovertemplate="%{text}<br>PC=(%{x:.2f},%{y:.2f},%{z:.2f})"))
    fig.update_layout(template="plotly_white", height=650,
        title=f"Token trajectories / fixed PCA basis / retained variance {sum(variance):.1%}",
        scene=dict(xaxis_title="PC1",yaxis_title="PC2",zaxis_title="PC3"))
    return fig


def generation_movie(runs):
    width = len(runs[-1]["ids"])
    ceiling = max(float(np.linalg.norm(r["residual"],axis=-1).max()) for r in runs)
    def heat(run):
        z = np.full((25,width), np.nan)
        z[:,:len(run["ids"])] = np.linalg.norm(run["residual"],axis=-1)
        return go.Heatmap(z=z,x=runs[-1]["labels"],y=run["stages"], zmin=0,zmax=ceiling,colorscale="Viridis",colorbar=dict(title="L2 norm"))
    frames=[go.Frame(name=str(i),data=[heat(r)]) for i,r in enumerate(runs)]
    fig=animate(go.Figure(data=[heat(runs[0])]),frames,"Autoregressive time: one new token per frame","generated tokens")
    fig.update_yaxes(autorange="reversed")
    return fig


def circuit_movie(runs, layer=0, head=0):
    if not (0 <= layer < 12 and 0 <= head < 12):
        raise ValueError("Layer/head must be in 0..11.")
    n=len(runs[-1]["ids"])
    score_bound=max(float(np.nanmax(np.abs(r["scores"][layer,head]))) for r in runs)
    write_bound=max(float(np.abs(r["head_writes"][layer,head]).max()) for r in runs)
    def data(r):
        size=len(r["ids"])
        score=np.full((n,n),np.nan); attention=score.copy(); write=np.full((n,768),np.nan)
        score[:size,:size]=r["scores"][layer,head]
        attention[:size,:size]=r["attention"][layer,head]
        write[:size]=r["head_writes"][layer,head]
        common=dict(y=runs[-1]["labels"],showscale=False)
        return [go.Heatmap(z=score,x=runs[-1]["labels"],zmin=-score_bound,zmax=score_bound,colorscale="RdBu",reversescale=True,**common),
                go.Heatmap(z=attention,x=runs[-1]["labels"],zmin=0,zmax=1,colorscale="Viridis",**common),
                go.Heatmap(z=write,x=np.arange(768),zmin=-write_bound,zmax=write_bound,colorscale="RdBu",reversescale=True,**common)]
    fig=make_subplots(rows=1,cols=3,subplot_titles=("QK scores / sqrt(64)","Attention probabilities","OV write: A @ V @ W_O"),horizontal_spacing=.09)
    for i,d in enumerate(data(runs[0])): fig.add_trace(d,row=1,col=i+1)
    frames=[go.Frame(name=str(i),data=data(r)) for i,r in enumerate(runs)]
    fig=animate(fig,frames,f"Layer {layer}, head {head}: QK / attention / OV over generation","generated tokens")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="source token",row=1,col=1)
    fig.update_xaxes(title_text="source token",row=1,col=2)
    fig.update_xaxes(title_text="residual coordinate",row=1,col=3)
    return fig


def source_contributions(run, layer, head, destination):
    if not 0 <= destination < len(run["ids"]):
        raise ValueError("Destination is outside this sequence.")
    contributions=run["attention"][layer,head,destination,:,None]*run["source_ov"][layer,head]
    np.testing.assert_allclose(contributions.sum(0),run["head_writes"][layer,head,destination],atol=2e-4,rtol=2e-4)
    fig=go.Figure(go.Heatmap(z=contributions,x=np.arange(768),y=run["labels"],colorscale="RdBu",zmid=0))
    fig.update_layout(template="plotly_white",height=500,title=f"L{layer} H{head}: each source's write to destination {destination}",xaxis_title="residual coordinate",yaxis_title="source token")
    return fig


@torch.inference_mode()
def logit_lens(model, run, tokenizer, k=8):
    residual=torch.as_tensor(run["residual"][:,-1],device=model.device)
    logits=model.lm_head(model.transformer.ln_f(residual))
    candidates=logits[-1].topk(k).indices
    np.testing.assert_allclose(cpu(logits[-1]),run["logits"][-1],atol=2e-4,rtol=2e-4)
    fig=go.Figure(go.Heatmap(z=cpu(logits[:,candidates]).T,x=run["stages"],y=[repr(tokenizer.decode([int(t)])) for t in candidates],colorscale="RdBu",zmid=0))
    fig.update_layout(title="Logit lens at the final input position (final-LN readout)",template="plotly_white",height=500,xaxis_title="depth",yaxis_title="candidate next token")
    return fig


def one_layer_demo():
    """Explicit pedagogical weights: NOT GPT-2 weights or a trained model."""
    vocab=["Alice","Bob","said","hello","goodbye"]
    e=np.eye(5); wu=np.eye(5); wq=np.zeros((5,2)); wk=np.zeros((5,2))
    wq[2,0]=3; wk[0,0]=3; wk[1,0]=2
    wv=np.eye(5); wo=np.zeros((5,5)); wo[0,3]=4; wo[1,4]=4
    ids=np.array([0,2]); x=e[ids]
    scores=(x@wq)@(x@wk).T/np.sqrt(2)
    scores[np.triu_indices(2,1)]=-np.inf
    a=np.exp(scores-np.max(scores,axis=-1,keepdims=True)); a/=a.sum(-1,keepdims=True)
    direct=x@wu; head=a@(x@wv@wo)@wu
    logits=(x+a@(x@wv@wo))@wu
    np.testing.assert_allclose(logits,direct+head)
    expanded_qk=e@wq@wk.T@e.T/np.sqrt(2)
    expanded_ov=e@wv@wo@wu
    np.testing.assert_allclose(head,a@expanded_ov[ids])
    fig=make_subplots(rows=1,cols=3,subplot_titles=("Expanded QK: destination -> source","Expanded OV: source -> output","Last-position logit decomposition"))
    fig.add_trace(go.Heatmap(z=expanded_qk,x=vocab,y=vocab,showscale=False),row=1,col=1)
    fig.add_trace(go.Heatmap(z=expanded_ov,x=vocab,y=vocab,showscale=False),row=1,col=2)
    for name,z in [("direct",direct[-1]),("head",head[-1])]:
        fig.add_trace(go.Bar(x=vocab,y=z,name=name),row=1,col=3)
    fig.update_layout(template="plotly_white",barmode="stack",height=450,title="One-layer attention-only: hand-built skip-trigram example")
    return fig,dict(vocab=vocab,input="Alice said",attention=a.tolist(),direct=direct.tolist(),head=head.tolist(),logits=logits.tolist(),max_decomposition_error=float(np.abs(logits-direct-head).max()))


def save_static(run, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output_dir=Path(output_dir); output_dir.mkdir(exist_ok=True,parents=True)
    fig,axes=plt.subplots(1,2,figsize=(16,7),constrained_layout=True)
    im=axes[0].imshow(np.linalg.norm(run["residual"],axis=-1),aspect="auto",cmap="viridis")
    axes[0].set(yticks=range(25),yticklabels=run["stages"],xticks=range(len(run["ids"])),xticklabels=run["labels"],title="Residual L2 norm",xlabel="token position")
    axes[0].tick_params(axis="x",rotation=90); fig.colorbar(im,ax=axes[0])
    norms=np.linalg.norm(run["updates"],axis=-1).mean(-1)
    axes[1].plot(range(12),norms[:,0],"o-",label="attention update (includes output bias)")
    axes[1].plot(range(12),norms[:,1],"s-",label="MLP update")
    axes[1].set(xlabel="layer (0-based)",ylabel="mean token L2 norm",title="How much each sublayer writes")
    axes[1].legend(); fig.savefig(output_dir/"residual-summary.png",dpi=140); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(17,5),constrained_layout=True)
    for ax,z,title in zip(axes,[run["scores"][0,0],run["attention"][0,0],run["head_writes"][0,0]],["L0 H0 QK scores","L0 H0 attention","L0 H0 OV writes (768 coordinates)"]):
        im=ax.imshow(z,aspect="auto",cmap="viridis"); ax.set_title(title); ax.set_ylabel("destination token"); fig.colorbar(im,ax=ax)
    fig.savefig(output_dir/"circuits-summary.png",dpi=140); plt.close(fig)


def figure_html(fig, include_plotlyjs="cdn"):
    # Plotly rejects an interrupted animation with undefined. Handle cancellation,
    # but retain actual errors, for both pause buttons and slider interruptions.
    script = """
    const graph = document.getElementById('{plot_id}');
    const animate = args => Plotly.animate(graph, ...args).catch(error => {
        if (error !== undefined) throw error;
    });
    graph.on('plotly_buttonclicked', event => {
        if (event.button.method === 'animate') animate(event.button.args);
    });
    graph.on('plotly_sliderchange', event => {
        if (event.step.method === 'animate') animate(event.step.args);
    });
    """
    return fig.to_html(full_html=False,include_plotlyjs=include_plotlyjs,
                       auto_play=False,post_script=script)


def save_report(figures, path):
    html=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>Mechanistic Interpretability Study</title><body><h1>GPT-2 Residual Stream Study</h1><p>Depth and generation time are separate axes. See the notebook for assumptions and interpretation.</p>']
    for i,fig in enumerate(figures):
        html.append(figure_html(fig,include_plotlyjs=True if i==0 else False))
    html.append("</body></html>")
    Path(path).write_text("\n".join(html),encoding="utf-8")


def metadata(model, prompt, runs, prefix_error):
    return dict(model=MODEL_ID,revision=REVISION,device=torch.cuda.get_device_name(),
        python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
        packages={p:importlib.metadata.version(p) for p in ["transformers","numpy","plotly","nbformat","nbclient"]},
        dtype=str(next(model.parameters()).dtype),seed=42,prompt=prompt,
        generation="greedy; no KV cache; full prefix recomputed",generated_text=runs[-1]["text"],
        generation_steps=len(runs)-1,prefix_invariance_max_abs_error=prefix_error,
        checks={key:max(r["errors"][key] for r in runs) for key in runs[0]["errors"]})
