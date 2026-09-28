"""Build the versioned notebook from readable, reviewable cell sources."""
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(source):
    cells.append(dict(cell_type="markdown", metadata={}, source=textwrap.dedent(source).strip()+"\n"))


def code(source):
    cells.append(dict(cell_type="code", metadata={}, source=textwrap.dedent(source).strip()+"\n", execution_count=None, outputs=[]))


md(r"""
# Week 1 · GPT-2의 Residual Stream과 Transformer Circuits

**스터디 범위:** *A Mathematical Framework for Transformer Circuits*의 도입부터
**One-Layer Attention-Only Transformers**까지. Two-Layer 및 induction-head 이론은 다음 주로 남긴다.

**이번 실험:** 실제 GPT-2 small (12 layers, 12 heads, $d_{model}=768$, $d_{head}=64$)에
문장을 넣고 768차원 상태가 어디에서 어떻게 바뀌는지 관찰한다. 별도의 작은 수식 예제로
one-layer attention-only 분해를 검증한다. 실제 모델을 one-layer 모델로 부르지 않는다.

**학습 목표**
1. residual stream을 각 위치의 벡터에 모듈들이 값을 더하는 공유 공간으로 설명한다.
2. QK의 정보 선택과 OV의 정보 변환을 구분한다.
3. 레이어 깊이, 문장 내 토큰 위치, 자기회귀 생성 시간을 구분한다.
4. 그림의 인상과 수치 검증, 인과적 주장을 구분한다.

권장 진행: 개념 20분 → residual 실험 25분 → QK/OV 25분 → one-layer 수식/토론 20분.
원문을 대체하는 번역문이 아니라 직접 실행하며 사용하는 실험 노트다.
""")

md(r"""
## 0. 실행 환경과 재현성

RTX 5090의 WSL Python 커널에서 **Restart Kernel → Run All**로 실행한다.
설치는 저장소 README 참고. GPU가 없으면 조용히 CPU로 바꾸지 않고 오류를 낸다.
모델은 eval/float32, greedy decoding, 고정 revision을 사용한다. 학습은 하지 않는다.

`PROMPT`와 `NEW_TOKENS`를 바꾸고 아래 셀부터 다시 실행하면 새 입력을 관찰할 수 있다.
GPT-2는 영어 중심 모델이고 BPE 토큰은 단어와 다르다. 한국어도 입력할 수 있지만
잘린 UTF-8 조각은 `�`로 표시될 수 있으므로 토큰 ID와 위치를 함께 확인한다.

GitHub 미리보기는 PNG와 수치 결과를 볼 수 있다. Play/슬라이더는 신뢰한 Jupyter 노트북
또는 `artifacts/week01-interactive.html`에서 동작한다. HTML은 Plotly를 포함해 오프라인에서도 열린다.
노트북 내 동적 출력은 Plotly CDN을 사용한다.
""")
code("""
from pathlib import Path
import sys, json
ROOT = Path.cwd()
if not (ROOT / "study.py").exists():
    ROOT = ROOT.parent
assert (ROOT / "study.py").exists(), "Run from the repository or notebooks directory."
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from IPython.display import display, HTML, Image
import study
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
PROMPT = "When Mary and John went to the store, John gave a drink to"
NEW_TOKENS = 6
LAYER = 0
HEAD = 0
model, tokenizer = study.load_model()
print(torch.cuda.get_device_name(), next(model.parameters()).dtype)
print(f"layers={model.config.n_layer}, heads={model.config.n_head}, residual={model.config.n_embd}")
""")
md(r"""
## 1. Residual stream은 어디에 있는가?

배치 축을 제외하면 $R\in\mathbb{R}^{T\times768}$. 각 행은 한 토큰 **위치**의 상태다.
초기 상태는 토큰 임베딩과 학습된 위치 임베딩의 합이다.

$$R_0=E[token]+P[position]$$
$$R_{\ell,mid}=R_{\ell,pre}+Attention_\ell(LN_{1,\ell}(R_{\ell,pre}))$$
$$R_{\ell,post}=R_{\ell,mid}+MLP_\ell(LN_{2,\ell}(R_{\ell,mid}))$$

블록 사이에는 $R_{\ell+1,pre}=R_{\ell,post}$이다. 마지막에는
$logits=LN_f(R_{11,post})W_U$. $LN_f$는 마지막 residual 업데이트가 아니라 읽기 전 정규화다.

이 노트북은 **초기 상태 1개 + 12블록 × (attention 이후, MLP 이후) = 25개 상태**를 저장한다.
Hugging Face의 `hidden_states[-1]`은 마지막 LayerNorm 이후라서 그대로 raw residual로
취급하면 안 된다. 여기서는 블록과 서브모듈 hook으로 직접 수집한다.

MLP는 위치별로 계산하고, attention은 다른 위치의 정보를 읽는다. 둘 다 같은 768차원 공간에
쓴다. 개별 좌표를 바로 “성별”, “이름” 같은 의미로 해석할 근거는 없다.
""")
code("""
runs, prefix_error = study.experiment(model, tokenizer, PROMPT, NEW_TOKENS)
run = runs[0]
print("Input:", run["text"])
print("Generated:", runs[-1]["text"])
print("Residual shape [stage, token, coordinate]:", run["residual"].shape)
for token_id, label in zip(run["ids"], run["labels"]):
    print(f"{token_id:5d}  {label}")
print("Numerical checks:", run["errors"])
""")
md(r"""
### 먼저 전체 크기를 본다

왼쪽 그림은 위치별 $\|R\|_2$, 오른쪽은 각 모듈이 **추가한 벡터**의 평균 크기다.
residual 자체의 크기와 업데이트 크기는 다르다. 큰 업데이트가 반드시 중요한 것도 아니다.
두 업데이트가 상쇄되거나 출력에 덜 민감한 방향으로 움직일 수 있다.
색 범위는 한 그림 안에서 고정된다. 첫 레이어/헤드 선택은 설명을 위한 기본값이며,
이 문장에 가장 중요한 회로를 발견했다는 뜻이 아니다.
""")
code("""
study.save_static(run, ARTIFACTS)
display(Image(filename=str(ARTIFACTS / "residual-summary.png")))
""")
md(r"""
### 768개 좌표 그대로 보기: 깊이 애니메이션

가로축은 768 residual 좌표, 세로축은 입력 토큰 위치다. Play 또는 depth 슬라이더로
초기 임베딩 → attention 덧셈 → MLP 덧셈을 따라간다. E는 초기 상태, A/M은 attention/MLP 이후다.
모든 프레임은 같은 색 범위를 쓴다. 큰 이상값 때문에 나머지가 안 보이지 않도록 색은
$sign(x)\log(1+|x|)$로 표시하고, **hover에는 변환 전 activation**을 표시한다.
`study.residual_movie(run, scale="raw")`로 선형 색상을 볼 수 있다. 값의 크기를 색 차이와
선형적으로 대응시키면 안 된다.

관찰 질문: 어느 층에서 변화가 큰가? 동일한 `John`도 위치에 따라 상태가 같은가?
768개 좌표의 패턴만으로 의미를 확정할 수 있는가?
""")
code("""
def show(fig):
    display(HTML(study.figure_html(fig)))
residual_fig = study.residual_movie(run)
show(residual_fig)
""")
md(r"""
### 공통 PCA 좌표계에서 토큰의 경로 보기

모든 깊이·생성 단계의 residual을 모아 **한 번만** PCA를 적합한다. 각 토큰의 선은 깊이에
따른 경로이고 hover는 서브레이어를 보여준다. 프레임마다 PCA를 새로 맞추면 축 자체가
회전하므로 이동을 비교할 수 없다. 여기서는 같은 중심과 기저를 쓴다.

PCA는 큰 분산을 보존하는 투영이다. 아래 분산 보존율이 낮으면 3D 그림에서 가까운 점이
768차원에서 가깝다고 말할 수 없다. PCA 축에 자동으로 의미 이름을 붙이지 않는다.
전체 궤적을 본 뒤 맞춘 좌표이므로 온라인 예측 분석도 아니다.
기본 입력에서는 첫 토큰의 큰 activation이 분산을 지배한다. 높은 분산 보존율만으로
다른 토큰의 세밀한 차이까지 잘 보인다고 결론내리지 않는다.
""")
code("""
mean, basis, variance = study.fit_projection(runs)
print("Explained variance per PC:", variance, "total:", variance.sum())
trajectory_fig = study.trajectory(run, mean, basis, variance)
show(trajectory_fig)
""")
md(r"""
## 2. '시간에 따라 변한다'는 말의 세 가지 의미

| 축 | 무엇이 변하는가? | 이 실험의 표시 |
|---|---|---|
| 깊이 $\ell$ | 동일 입력의 상태가 서브레이어를 통과 | 25개 residual 단계 |
| 위치 $t$ | 동일 문장의 서로 다른 토큰 | heatmap 행/열과 토큰 라벨 |
| 생성 step $s$ | 접두사에 새 토큰이 추가 | 다음 애니메이션의 프레임 |

causal mask 때문에 뒤에 토큰을 추가해도 **기존 위치의 상태는 변하지 않는다**.
수치 계산 오차를 제외하면 이전 프레임의 영역이 고정이고 새 열만 나타나야 한다.
새로 생긴 마지막 위치의 벡터와 이전 마지막 위치의 벡터를 비교할 때는 서로 다른 위치라는
점을 기억한다. 이것은 학습 시간이나 wall-clock 속도 측정이 아니다.

교육용으로 매 step 전체 prefix를 다시 계산한다(`use_cache=False`). 실제 서빙의 KV cache
최적화와 구별하기 위한 선택이며, 짧은 문장만 사용한다. greedy가 EOS를 골라도 이 실험은
지정한 step 수만큼 이어서 관측한다.
""")
code("""
print("Old-prefix residual max absolute change:", prefix_error)
for step, item in enumerate(runs):
    print(step, repr(item["text"]))
time_fig = study.generation_movie(runs)
show(time_fig)
""")
md(r"""
## 3. Attention을 QK와 OV로 분리하기

이 노트북은 **행벡터** 표기다. 원문의 열벡터 표기와 행렬 곱 순서가 반대일 수 있다.
한 head에서 $X=LN_1(R)$라고 두면:

$$Q=XW_Q+b_Q,\quad K=XW_K+b_K,\quad V=XW_V+b_V$$
$$S_{ij}=Q_iK_j^T/\sqrt{64},\quad A=softmax(S+M)$$
$$O^h_i=\sum_{j\le i} A^h_{ij}(V^h_jW^h_O)$$

행 $i$는 정보를 받는 **destination**, 열 $j$는 정보를 주는 **source**다.
미래 위치는 mask로 제외된다. QK는 *어디를 읽을지*, OV는 *읽은 것을 어떤 벡터로 쓸지*를
결정한다. attention 가중치만 보아서는 출력의 크기·방향·부호를 알 수 없다.

$W_{QK}=W_QW_K^T$와 $W_{OV}=W_VW_O$는 각각 $768\times768$의 **고정 가중치 곱**이다.
아래 움직이는 그림은 그 가중치 자체가 아니라 입력에 따른 QK 점수와 OV 쓰기 벡터다.
GPT-2의 bias 때문에 실제 식은 순수한 두 행렬 곱보다 항이 더 있다.

모든 head 출력을 더한 뒤 attention 출력 bias를 **한 번** 더한다:
$Attention(X)=\sum_h O^h+b_O$. $b_V$는 각 head의 OV에 포함하고 $b_O$는 head별로 중복 배분하지 않는다.
dropout은 eval 상태에서 꺼져 있다.
""")
code("""
display(Image(filename=str(ARTIFACTS / "circuits-summary.png")))
circuit_fig = study.circuit_movie(runs, LAYER, HEAD)
show(circuit_fig)
""")
md(r"""
### 레이어·헤드 선택과 source별 OV 기여

아래 선택기는 GPU 계산 없이 저장된 activation을 사용한다. 12×12 조합을 바꿔 보고
`Render selected head` 버튼을 누른 후 생성 슬라이더를 움직인다.
첫 두 그림의 색 범위는 각각 고정된 QK 범위와 $[0,1]$,
세 번째는 고정된 OV 범위다. 서로 다른 단위의 패널 색을 직접 비교하지 않는다.

source별 그림은 마지막 destination에 각 source가 쓴 $A_{ij}V_jW_O$다.
source 축으로 합하면 해당 head의 출력 벡터가 되어야 한다. 음의 값도 정상이다.
""")
code("""
import ipywidgets as widgets
def inspect_head(layer=0, head=0):
    show(study.circuit_movie(runs, layer, head))
    show(study.source_contributions(runs[0], layer, head, len(runs[0]["ids"])-1))
controls = widgets.interactive(inspect_head, {"manual": True, "manual_name": "Render selected head"},
    layer=widgets.IntSlider(min=0,max=11,value=LAYER,description="Layer",continuous_update=False),
    head=widgets.IntSlider(min=0,max=11,value=HEAD,description="Head",continuous_update=False))
display(controls)
""")
md(r"""
### 고정 가중치 곱과 입력 의존 패턴 비교

가중치 곱은 이 생성 실험 동안 변하지 않는다. 아래는 선택된 head의 QK/OV를 직접
구성하고 rank의 상한이 head dimension 64임을 보여준다. bias와 LayerNorm을 생략한
이 곱을 앞의 실제 activation heatmap과 혼동하지 않는다.
""")
code("""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
attn = model.transformer.h[LAYER].attn
weight = study.cpu(attn.c_attn.weight)
wq, wk, wv = np.split(weight, 3, axis=1)
sl = slice(HEAD*64, (HEAD+1)*64)
wo = study.cpu(attn.c_proj.weight)[sl]
wqk = wq[:,sl] @ wk[:,sl].T / np.sqrt(64)
wov = wv[:,sl] @ wo
weight_fig = make_subplots(rows=1, cols=2, subplot_titles=("W_Q W_K^T / sqrt(64)", "W_V W_O"))
for column, matrix in enumerate([wqk, wov], start=1):
    # 768x768 matrices are shown at full resolution, with independent scales.
    weight_fig.add_trace(go.Heatmap(z=matrix,colorscale="RdBu",zmid=0,showscale=False),row=1,col=column)
weight_fig.update_layout(height=550,title=f"Fixed weights: layer {LAYER}, head {HEAD}")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
for ax, matrix, title in zip(axes,[wqk,wov],["W_Q W_K^T / sqrt(64)","W_V W_O"]):
    bound = np.abs(matrix).max()
    im = ax.imshow(matrix,cmap="RdBu_r",vmin=-bound,vmax=bound)
    ax.set(title=title,xlabel="output/key coordinate",ylabel="input/query coordinate")
    fig.colorbar(im,ax=ax)
fig.savefig(ARTIFACTS / "fixed-weights.png",dpi=150)
plt.close(fig)
display(Image(filename=str(ARTIFACTS / "fixed-weights.png")))
# Optional interactive rendering of all weights: show(weight_fig)
print("Both maps factor through a 64-dimensional head space; rank <=64.")
""")
md(r"""
## 4. Residual을 어휘 공간에서 읽기: logit lens

각 단계의 마지막 위치에 최종 $LN_f$와 unembedding을 적용한다.
마지막 단계는 실제 모델 logits와 일치해야 한다. 중간 단계에는 이 readout을 위해
학습된 보장이 없으므로 **진단용 탐침**이다. 모델이 그 순간 해당 단어를 확신했다고
해석하지 않는다. 후보는 최종 단계 top-8로 고정한다. 표는 확률이 아닌 logits다.

특히 최종 LayerNorm의 평균·분산이 입력에 의존하므로, head의 $O^hW_U$를
실제 최종 logits에 대한 정확한 가산 기여라고 부를 수 없다.
""")
code("""
lens_fig = study.logit_lens(model, run, tokenizer)
show(lens_fig)
""")
md(r"""
## 5. Zero-layer에서 One-Layer Attention-Only까지

먼저 위치 임베딩, bias, LayerNorm, MLP를 생략한 단순 모델을 생각한다.
토큰 one-hot 행렬을 $T$, 임베딩을 $W_E$라고 하면 $X=TW_E$.
zero-layer의 logits는 $TW_EW_U$이므로 현재 토큰에서 다음 토큰으로의 직접 경로다.

한 층의 attention만 넣으면:

$$logits=TW_EW_U+\sum_h A^hTW_EW_V^hW_O^hW_U$$

직접 경로와 head 경로의 합이라는 것이 핵심이다. vocabulary 공간으로 펼치면
$C_{QK}=W_EW_QW_K^TW_E^T/\sqrt{d_h}$,
$C_{OV}=W_EW_VW_OW_U$. 전자는 destination/source 조합의 점수,
후자는 source/output 조합의 logit 변화를 표현한다.

`[source] ... [destination] [output]`의 세 토큰을 생각하면 QK가 source를 선택하고
OV가 output 후보를 바꾼다. 원문은 이를 skip-trigram 관점으로 설명한다.
softmax 분모는 prefix 전체에 의존하므로 독립적인 trigram 빈도표와 정확히 같지는 않다.

아래 작은 모델은 **설명용으로 직접 정한 가중치**다. GPT-2를 자르거나 학습한 모델이 아니다.
`Alice said`에서 `said`가 `Alice`를 읽고 `hello` logit을 올리도록 설계했다.
이를 통해 분해의 대수적 정확성을 확인할 수 있지만 GPT-2가 동일한 회로를 학습했다는
증거는 얻을 수 없다. 원문 실험의 재현이나 학습 결과 재현을 주장하지 않는다.
""")
code("""
toy_fig, toy = study.one_layer_demo()
show(toy_fig)
print(json.dumps(toy, indent=2))
assert toy["max_decomposition_error"] < 1e-12
""")
md(r"""
### GPT-2에 적용할 때 달라지는 부분

| 단순 one-layer 모델 | 실제 GPT-2 실험 |
|---|---|
| attention만 한 층 | attention+MLP 블록 12개 |
| token embedding만 가정 | token+position embedding |
| 선형 경로의 합 | LayerNorm, MLP 비선형성이 포함됨 |
| bias 생략 | Q/K/V 및 output bias 포함 |
| 작은 어휘로 전체 회로 확인 | 50,257 어휘: 전체 어휘 쌍 행렬은 만들지 않음 |

GPT-2 첫 블록의 attention만 관찰해도 원문의 순수한 one-layer attention-only 모델과 같지 않다.
여기서는 전체 GPT-2 안의 실제 연산을 hook으로 검증하고, 대수적 단순 모델은 별도로 검증한다.
다층 head composition의 해석은 이번 주 범위 밖이다.
""")
md(r"""
## 6. 스터디 실습과 해석의 한계

1. `PROMPT = "The capital of France is"`로 바꾼 뒤 실행한다. attention/MLP 중 어느 단계에서
   최종 후보의 logit이 바뀌는가? 변화 크기와 정답 확률 변화는 같은가?
2. `John`을 `Mary`로 교체한다. BPE 길이가 달라지면 같은 위치 비교가 어긋나므로 토큰 표부터 확인한다.
3. 같은 head에서 QK가 큰 source와 OV 기여가 큰 source가 같은지 본다. attention만으로 설명이 충분한가?
4. 생성 애니메이션에서 이전 위치가 고정되는 이유를 causal mask로 설명한다.
5. toy 모델의 `Alice`와 `Bob`을 바꾸려면 `study.one_layer_demo`의 ids를 어떻게 바꾸어야 하는가?
   직접 경로와 head 경로를 따로 예측한 후 실행해 본다.

**이번에 얻는 것은 관측과 대수 검증이다.** activation patching/ablation으로 인과성을 검증하지
않았고, 한 문장의 한 head 패턴은 일반적인 기능을 입증하지 않는다. PCA·norm·logit lens는
서로 다른 요약이며 어느 하나도 모델의 의미를 완전히 설명하지 않는다.
""")
md(r"""
## 7. 실험 2: Q, K, V, O와 QK, OV의 유효 랭크

**질문:** head dimension은 64지만 실제 가중치의 특이값은 64개 방향에 고르게 퍼져 있는가?
12개 레이어 × 12개 헤드 **전체 144개**를 측정한다. 여기서 Q/K/V/O는 activation이 아니라
각 head의 $W_Q,W_K,W_V,W_O$를 뜻한다. bias와 LayerNorm은 이 선형 가중치 실험에서 제외한다.

| 행렬 (행벡터 표기) | 모양 | 랭크의 상한 |
|---|---|---|
| $W_Q,W_K,W_V$ | $768\times64$ | 64 |
| $W_O$ (해당 head의 output projection 블록) | $64\times768$ | 64 |
| $W_{QK}=W_QW_K^T/\sqrt{64}$ | $768\times768$ | 64 |
| $W_{OV}=W_VW_O$ | $768\times768$ | 64 |

전체 12-head output projection은 $768\times768$이고 랭크가 64에 제한되지 않는다.
이번의 O는 **head별 64개 행**이다. 마찬가지로 QK는 softmax 이후 attention 행렬이 아니다.
softmax는 비선형이므로 attention 행렬 자체에 같은 rank 상한을 적용할 수 없다.

짧은 문장에서 $Q=XW_Q$의 랭크를 측정하면 $rank(Q)\le\min(T,64)$이다.
토큰 수가 14개라서 64보다 작아지는 것을 “모델이 저차원을 학습했다”는 증거로 삼으면 안 된다.
따라서 먼저 문장에 무관한 **가중치 스펙트럼**을 관찰한다. 생성 중 가중치는 변하지 않는다.
""")
md(r"""
### 랭크를 한 가지 숫자로 혼동하지 않기

$\sigma_1\ge\cdots\ge\sigma_{64}\ge0$를 특이값이라고 할 때:

- **수치적 랭크:** $\#\{\sigma_i>\max(m,n)\epsilon\sigma_1\}$.
  float64/float32 epsilon 기준을 모두 보고한다. 후자는 같은 float64 스펙트럼에 더 큰
  임계값을 적용한 민감도 분석이며, float32 SVD를 다시 실행한 결과가 아니다.
- **Entropy effective rank (주 지표):** $p_i=\sigma_i/\sum_j\sigma_j$,
  $r_{eff}=\exp(-\sum_i p_i\log p_i)$. 제곱 특이값으로 정의한 entropy와 구별한다.
- **Stable rank:** $\sum_i\sigma_i^2/\sigma_1^2$.
- **Energy participation ratio:** $(\sum_i\sigma_i^2)^2/\sum_i\sigma_i^4$.
- **k90 / k99:** 누적 $\sigma_i^2$의 90% / 99%를 보존하는 최소 방향 수.
  k99는 최적 truncated SVD의 상대 Frobenius 오차가 0.1 이하라는 의미다.

동일한 크기의 특이값 64개면 effective rank=64, 한 방향뿐이면 1이다.
조금이라도 불균일하면 64보다 작으므로 **64 미만이라는 사실만으로 강한 저랭크성을
주장하지 않는다.** 중앙값·범위·스펙트럼과 k99를 함께 읽는다. 영행렬은 편의상 모든 지표를 0으로 정의한다.

GPU에서 가중치를 float64로 변환해 계산한다. 원래 float32 가중치에 없던 정밀도가 생기는 것은 아니다.
QK/OV는 thin QR을 이용해 동일한 비영 특이값을 가진 $64\times64$ 행렬로 줄여 계산한다.
$A=U_AR_A$, $B^T=U_BR_B$이면 $AB=U_A(R_AR_B^T)U_B^T$이다.
레이어 0/6/11, 헤드 0에서는 전체 $768\times768$ SVD와 직접 비교해 이 계산을 검증한다.

Effective rank 정의: [Roy & Vetterli (2007)](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/a5p-h05.pdf).
""")
code("""
import rank_analysis
from IPython.display import Markdown
print(rank_analysis.check_metric_definitions())
rank_result = rank_analysis.analyze(model)
rank_analysis.save(rank_result, ARTIFACTS)
display(Markdown(rank_analysis.summary_markdown(rank_result)))
print("Measured:",len(rank_result["records"]),"matrices; 144 heads x 6 matrix types")
print("Direct full-matrix SVD checks:",json.dumps(rank_result["validation"],indent=2))
""")
md(r"""
### 모든 레이어·헤드 비교

여섯 패널은 **같은 0–64 색 범위**를 쓴다. 드롭다운에서 effective rank, stable rank,
k99 또는 수치 랭크를 바꿀 수 있다. 어두운 head는 그 지표에서 더 적은 방향에 집중된다.
stable rank가 작아도 꼬리의 여러 방향이 합쳐 상당한 에너지를 가질 수 있으므로 k99도 확인한다.
""")
code("""
display(Image(filename=str(ARTIFACTS / "effective-rank.png")))
rank_fig = rank_analysis.rank_heatmap(rank_result)
show(rank_fig)
rank_spectrum_fig = rank_analysis.spectrum_figure(rank_result,LAYER,HEAD)
show(rank_spectrum_fig)
""")
code("""
def inspect_rank(layer=0,head=0):
    show(rank_analysis.spectrum_figure(rank_result,layer,head))
    rows=[r for r in rank_result["records"] if r["layer"]==layer and r["head"]==head]
    for row in rows:
        print(f"{row['matrix']:>2}: effective={row['effective_rank']:.2f}, "
              f"stable={row['stable_rank']:.2f}, k99={row['k99']}, "
              f"numerical(fp64/fp32)={row['rank_fp64']}/{row['rank_fp32']}")
display(widgets.interactive(inspect_rank,{"manual":True,"manual_name":"Inspect spectrum"},
    layer=widgets.IntSlider(min=0,max=11,value=LAYER,description="Layer",continuous_update=False),
    head=widgets.IntSlider(min=0,max=11,value=HEAD,description="Head",continuous_update=False)))
""")
md(r"""
### 계산 결과에서 직접 답하기

다음 표는 각 행렬에서 effective rank가 64 미만인 헤드 수를 출력한다. 수치 랭크와 비교하면
“64개 독립 방향은 남아 있지만 세기가 불균일한가?”를 판단할 수 있다.
QK/OV가 개별 행렬보다 더 집중되는지도 **동일 head의 짝**으로 비교한다.
""")
code("""
for name in rank_analysis.KINDS:
    s=rank_result["summary"][name]
    print(f"{name:>2}: eRank < 64: {s['effective_below_64_count']}/144; "
          f"eRank < 32: {s['effective_below_32_count']}/144; "
          f"k99 < 64: {s['k99_below_64_count']}/144")
paired={(r["layer"],r["head"],r["matrix"]):r for r in rank_result["records"]}
for product,a,b in [("QK","Q","K"),("OV","V","O")]:
    delta=np.array([paired[l,h,product]["effective_rank"]-
                    min(paired[l,h,a]["effective_rank"],paired[l,h,b]["effective_rank"])
                    for l in range(12) for h in range(12)])
    print(f"{product}: eRank(product) < min(eRank({a}), eRank({b})): "
          f"{(delta < 0).sum()}/144; median paired difference={np.median(delta):.2f}")
""")
md(r"""
### 이번 체크포인트에서 관찰한 결과

고정된 GPT-2 revision에서 864개 행렬의 수치 랭크는 **두 epsilon 기준 모두 64**였다.
entropy effective rank는 모두 64 미만이지만, 중앙값은 Q=62.81, K=62.78, V=63.11,
O=62.98, QK=61.04, OV=61.86이다. 따라서 대부분의 head를 아주 작은 차원으로
줄일 수 있다고 결론내릴 근거는 약하다. k99 중앙값도 각각 62, 62, 63, 63, 61, 63이다.

반면 다음 head들은 눈에 띄게 집중되어 있다. 인덱스는 모두 **0부터 시작**한다.

| 행렬 | layer / head | effective rank | stable rank | k99 |
|---|---|---:|---:|---:|
| QK | L1 H8 | 28.35 | 4.43 | 27 |
| QK | L1 H9 | 30.30 | 2.59 | 31 |
| OV | L11 H8 | 18.80 | 1.05 | 20 |

특히 L11 H8의 OV는 stable rank가 약 1.05이므로 첫 방향이 제곱 특이값 에너지의
약 $1/r_{stable}\approx95.4\%$를 차지하지만, **99%까지 보존하려면 20개 방향**이 필요하다.
“거의 rank 1”이라는 말이 어떤 오차 기준인지 명시해야 하는 좋은 예다.
위 스펙트럼 선택기에서 이 head들을 직접 비교한다. 이 수치는 고정 revision의 측정 결과이며,
다른 체크포인트로 코드를 바꾸면 위의 실행 결과를 기준으로 다시 해석해야 한다.

### 무엇까지 해석할 수 있는가?

수치 랭크가 64인데 effective rank가 낮다면, 이 정의에서 스펙트럼의 집중을 관찰한 것이다.
이는 정확한 rank deficiency나 head dimension을 줄여도 성능이 유지된다는 증명이 아니다.
압축 가능성은 별도의 truncated-SVD 개입과 출력/성능 평가로 확인해야 한다.
가중치의 작은 방향도 입력 분포나 downstream readout에 따라 중요할 수 있다.

또한 Q/K 개별 유효 랭크는 head 내부의 일반적인 가역 기저 변환에 불변하지 않다.
$W_Q\to W_QS$, $W_K\to W_KS^{-T}$이면 QK는 그대로다.
마찬가지로 $W_V\to W_VS$, $W_O\to S^{-1}W_O$이면 OV는 그대로다.
그래서 개별 Q/K/V/O와 **합성된 QK/OV를 함께** 살핀다. 직교 기저 변환에서는
특이값이 보존되지만 일반적인 가역 변환에서는 그렇지 않다.

보통의 algebraic rank에는 $rank(AB)\le\min(rank(A),rank(B))$가 성립한다.
같은 부등식을 entropy effective rank에 자동으로 적용하지 않는다. 위 짝 비교는 실제 관측이다.
이 실험에는 무작위 초기화나 학습 전후 대조군이 없으므로 집중이 학습 때문에 생겼다고
단정할 수도 없다. raw 값은 `artifacts/effective-rank.csv`, 전체 스펙트럼은 JSON에 저장한다.
""")
code("""
report_figures = [residual_fig, trajectory_fig, time_fig, circuit_fig,
                   study.source_contributions(run,LAYER,HEAD,len(run["ids"])-1),lens_fig,toy_fig,
                   rank_fig,rank_spectrum_fig]
study.save_report(report_figures, ARTIFACTS / "week01-interactive.html")
evidence = study.metadata(model,PROMPT,runs,prefix_error)
evidence["pca_explained_variance"] = variance.tolist()
evidence["toy_decomposition_error"] = toy["max_decomposition_error"]
(ARTIFACTS / "validation.json").write_text(json.dumps(evidence,indent=2),encoding="utf-8")
np.savez_compressed(ARTIFACTS / "prompt-activations.npz",
    residual=run["residual"],attention=run["attention"],qk_scores=run["scores"],
    head_writes=run["head_writes"],source_ov=run["source_ov"],token_ids=run["ids"])
print(json.dumps(evidence,indent=2))
print("Saved offline interactive report and validation evidence to",ARTIFACTS)
""")
md(r"""
## 참고 자료와 다음 주

- Elhage et al. (2021), [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html).
  이번 주는 **One-Layer Attention-Only Transformers** 섹션 끝까지.
- [Transformer Circuits thread](https://transformer-circuits.pub/): 이후 아티클별 실험의 기준.
- [GPT-2 모델 카드](https://huggingface.co/openai-community/gpt2)와
  [고정 revision 설정](https://huggingface.co/openai-community/gpt2/blob/607a30d783dfa663caf39e06633721c8d4cfcd7e/config.json).
- [Hugging Face GPT-2 구현 (v4.57.6)](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/gpt2/modeling_gpt2.py):
  pre-LN, Conv1D 가중치 방향, head 합산과 bias를 확인하는 구현 근거.

다음 주 후보: two-layer composition을 위한 작은 attention-only 모델과 인과적 개입 실험.
""")

notebook=dict(cells=cells,metadata={"kernelspec":{"display_name":"Python 3 (GPU study)","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.10"}},nbformat=4,nbformat_minor=5)
for index,cell in enumerate(cells):
    cell["id"]=f"week01-{index:02d}"
path=ROOT/"notebooks"/"01_residual_stream_qk_ov.ipynb"
path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
print(path)
