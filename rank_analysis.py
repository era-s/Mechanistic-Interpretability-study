"""Experiment 2: head-wise weight spectra, computed in float64 on the GPU."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

KINDS = ("Q", "K", "V", "O", "QK", "OV")
METRICS = ("effective_rank", "stable_rank", "energy_pr", "k90", "k99", "rank_fp64", "rank_fp32")


def spectrum_metrics(s, shape):
    s = np.asarray(s, dtype=np.float64)
    if s.ndim != 1 or not np.isfinite(s).all() or np.any(s < 0):
        raise ValueError("Expected finite, nonnegative singular values")
    s = np.sort(s)[::-1]
    if not len(s) or s[0] == 0:
        return dict.fromkeys(METRICS, 0)
    scaled = s / s[0]
    p = scaled / scaled.sum()
    p = p[p > 0]
    energy = scaled**2
    cumulative = np.cumsum(energy) / energy.sum()
    return dict(
        effective_rank=float(np.exp(-np.sum(p*np.log(p)))),
        stable_rank=float(energy.sum()),
        energy_pr=float(energy.sum()**2/np.sum(energy**2)),
        k90=int(np.searchsorted(cumulative, .90)+1),
        k99=int(np.searchsorted(cumulative, .99)+1),
        rank_fp64=int(np.count_nonzero(scaled > max(shape)*np.finfo(np.float64).eps)),
        rank_fp32=int(np.count_nonzero(scaled > max(shape)*np.finfo(np.float32).eps)),
    )


def product_spectrum(left, right):
    """Exact nonzero spectrum of [m,r] @ [r,n] via a small QR core."""
    _, rleft = torch.linalg.qr(left, mode="reduced")
    _, rright = torch.linalg.qr(right.T, mode="reduced")
    return torch.linalg.svdvals(rleft @ rright.T, driver="gesvd")


@torch.inference_mode()
def analyze(model):
    if model.device.type != "cuda":
        raise ValueError("Run this study on the GPU kernel")
    records, spectra, checks = [], {}, []
    for layer, block in enumerate(model.transformer.h):
        q, k, v = block.attn.c_attn.weight.double().split(768, dim=1)
        out = block.attn.c_proj.weight.double()
        for head in range(12):
            sl = slice(head*64, (head+1)*64)
            matrices = dict(Q=q[:,sl], K=k[:,sl], V=v[:,sl], O=out[sl,:])
            singular = {name:torch.linalg.svdvals(a,driver="gesvd") for name,a in matrices.items()}
            singular["QK"] = product_spectrum(matrices["Q"],matrices["K"].T) / 8.
            singular["OV"] = product_spectrum(matrices["V"],matrices["O"])
            # Check the optimized calculation against explicit 768x768 SVDs.
            if layer in (0,6,11) and head == 0:
                for name, full in [("QK",matrices["Q"]@matrices["K"].T/8.),
                                   ("OV",matrices["V"]@matrices["O"])]:
                    direct=torch.linalg.svdvals(full,driver="gesvd")
                    torch.testing.assert_close(singular[name],direct[:64],atol=1e-10,rtol=1e-9)
                    tail_ratio=float(direct[64:].max()/direct[0])
                    assert tail_ratio < 1e-12
                    checks.append(dict(layer=layer,head=head,matrix=name,
                        max_spectrum_difference=float((singular[name]-direct[:64]).abs().max()),
                        omitted_tail_relative_max=tail_ratio))
            for name, s in singular.items():
                values=s.cpu().numpy()
                spectra[f"{layer}/{head}/{name}"]=values.tolist()
                shape=tuple(matrices[name].shape) if name in matrices else (768,768)
                metrics=spectrum_metrics(values,shape)
                assert 1-1e-10 <= metrics["effective_rank"] <= 64+1e-10
                assert 1 <= metrics["k90"] <= metrics["k99"] <= 64
                records.append(dict(layer=layer,head=head,matrix=name,rows=shape[0],cols=shape[1],**metrics))
    summary={}
    for name in KINDS:
        group=[r for r in records if r["matrix"]==name]
        summary[name]={metric:dict(min=float(min(r[metric] for r in group)),
            median=float(np.median([r[metric] for r in group])),
            max=float(max(r[metric] for r in group))) for metric in METRICS}
        summary[name]["effective_below_64_count"]=sum(r["effective_rank"] < 64-1e-6 for r in group)
        summary[name]["effective_below_32_count"]=sum(r["effective_rank"] < 32 for r in group)
        summary[name]["k99_below_64_count"]=sum(r["k99"] < 64 for r in group)
    return dict(records=records,spectra=spectra,summary=summary,validation=checks,
        metadata=dict(scope="144 individual heads; weight matrices, excluding biases and LayerNorm",
            svd_dtype="float64",device=torch.cuda.get_device_name(model.device),
            numerical_threshold="max(rows,cols) * dtype_epsilon * sigma_max",
            effective_rank="exp(entropy(sigma / sum(sigma)))",
            product_method="thin QR factors followed by 64x64 SVD; exact nonzero spectrum",
            qk_scale="1/sqrt(64); scale does not affect rank metrics",
            zero_matrix_convention=0))


def metric_grids(result, metric):
    grids={name:np.empty((12,12)) for name in KINDS}
    for row in result["records"]:
        grids[row["matrix"]][row["layer"],row["head"]]=row[metric]
    return grids


def rank_heatmap(result):
    fig=make_subplots(rows=2,cols=3,subplot_titles=KINDS,vertical_spacing=.16)
    for i,(name,z) in enumerate(metric_grids(result,"effective_rank").items()):
        fig.add_trace(go.Heatmap(z=z,x=list(range(12)),y=list(range(12)),coloraxis="coloraxis",
            hovertemplate=f"{name}<br>layer %{{y}}, head %{{x}}<br>value %{{z:.3f}}<extra></extra>"),row=i//3+1,col=i%3+1)
    buttons=[dict(label=m,method="update",args=[{"z":list(metric_grids(result,m).values())},
        {"title.text":f"Experiment 2 / {m} / per-head ceiling = 64"}]) for m in METRICS]
    fig.update_layout(template="plotly_white",height=720,
        title="Experiment 2 / effective_rank / per-head ceiling = 64",
        coloraxis=dict(colorscale="Viridis",cmin=0,cmax=64,colorbar=dict(title="rank / count")),
        updatemenus=[dict(buttons=buttons,x=1,y=1.16,xanchor="right")],margin=dict(t=140,b=60))
    fig.update_xaxes(title="head",dtick=1)
    fig.update_yaxes(title="layer",dtick=1,autorange="reversed")
    return fig


def spectrum_figure(result,layer=0,head=0):
    fig=make_subplots(rows=1,cols=2,subplot_titles=("Relative singular values (log scale)","Cumulative squared-singular-value energy"))
    colors=("#0072B2","#D55E00","#009E73","#CC79A7","#7A5D00","#222222")
    for name,color in zip(KINDS,colors):
        s=np.asarray(result["spectra"][f"{layer}/{head}/{name}"])
        energy=np.cumsum(s*s)/np.sum(s*s)
        fig.add_trace(go.Scatter(x=np.arange(1,65),y=s/s[0],name=name,legendgroup=name,line=dict(color=color)),row=1,col=1)
        fig.add_trace(go.Scatter(x=np.arange(1,65),y=energy,name=name,legendgroup=name,showlegend=False,line=dict(color=color)),row=1,col=2)
    fig.add_hline(y=.99,line_dash="dash",row=1,col=2)
    fig.update_yaxes(type="log",title="sigma_i / sigma_1",row=1,col=1)
    fig.update_yaxes(range=[0,1.01],title="retained energy",row=1,col=2)
    fig.update_xaxes(title="number of singular directions",range=[1,64])
    fig.update_layout(template="plotly_white",height=450,title=f"Experiment 2 / layer {layer}, head {head}")
    return fig


def summary_markdown(result):
    lines=["| 행렬 | effective rank 중앙값 [최소, 최대] | stable rank 중앙값 | k99 중앙값 | fp64 수치 랭크 범위 | eRank < 32 헤드 |",
           "|---|---:|---:|---:|---:|---:|"]
    for name in KINDS:
        s=result["summary"][name]; e=s["effective_rank"]; r=s["rank_fp64"]
        lines.append(f"| {name} | {e['median']:.2f} [{e['min']:.2f}, {e['max']:.2f}] | {s['stable_rank']['median']:.2f} | {s['k99']['median']:.1f} | {r['min']:.0f}–{r['max']:.0f} | {s['effective_below_32_count']}/144 |")
    return "\n".join(lines)


def save(result,output_dir):
    import matplotlib.pyplot as plt
    output_dir=Path(output_dir)
    (output_dir/"effective-rank.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    with (output_dir/"effective-rank.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(result["records"][0]))
        writer.writeheader(); writer.writerows(result["records"])
    fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    for ax,(name,z) in zip(axes.flat,metric_grids(result,"effective_rank").items()):
        im=ax.imshow(z,vmin=0,vmax=64,cmap="viridis")
        ax.set(title=name,xlabel="head",ylabel="layer",xticks=range(12),yticks=range(12))
    fig.colorbar(im,ax=axes.ravel().tolist(),label="entropy effective rank (ceiling 64)")
    fig.savefig(output_dir/"effective-rank.png",dpi=140)
    plt.close(fig)


def check_metric_definitions():
    np.testing.assert_allclose(spectrum_metrics(np.ones(64),(768,64))["effective_rank"],64)
    for metric in METRICS:
        assert spectrum_metrics(np.r_[1.,np.zeros(63)],(768,64))[metric]==1
        assert spectrum_metrics(np.zeros(64),(768,64))[metric]==0
    s=np.geomspace(1,.001,64)
    first=spectrum_metrics(s,(768,64)); scaled=spectrum_metrics(s*7,(768,64))
    for metric in METRICS:
        np.testing.assert_allclose(first[metric],scaled[metric])
    return "flat spectrum, rank-one, zero, and scale invariance: passed"
