"""[A] DeBERTa-v3-base 기본형과 주제 적대 학습 비교 실험 (작업 A-5, A-6).

사용법 (저장소 최상위 폴더에서, GPU가 켜진 Kaggle/Colab 권장):
    python experiments/A_deberta.py                       # λ=0(기본형), λ=0.1(적대 학습)을 val로 비교
    python experiments/A_deberta.py --lambdas 0 0.1 0.3   # 적대 강도를 더 넣어 비교 (선택)
    python experiments/A_deberta.py --final               # 설정 고정 뒤, 저장된 체크포인트로 test 한 번 평가

구조:
    글 → DeBERTa 인코더 → 평균 풀링 → 문서 벡터 z
                                       ├─ 작성자 분류 헤드 (18개 모델)        : 정답을 잘 맞히도록 학습
                                       └─ GRL → 주제 분류 헤드 (train 질문 105개) : 주제를 잘 맞히도록 학습
    GRL(Gradient Reversal Layer)은 앞으로 갈 때는 값을 그대로 넘기고, 역전파 때는 기울기에 -λ를 곱한다.
    그래서 주제 헤드는 주제를 맞히려 하고, 인코더는 반대로 주제를 맞히기 어려운 표현을 만들도록 학습된다.
    λ=0이면 인코더에 주제 신호가 전혀 전달되지 않으므로 기본형(작성자 분류만 학습)과 같다.

따르는 규칙 (docs/next-tasks-A-B-C-D.md):
- A의 분할을 그대로 쓰고, train으로만 학습한다. val은 에폭 선택(조기 종료)과 조건 비교에만 쓴다.
- 주제 라벨은 train의 prompt_id를 대용으로 쓴다. 실제 의미 주제와 같지 않다.
- val 질문 ID를 train 주제 분류기의 정답으로 평가하지 않는다.
  주제 정보 진단(probe)은 같은 질문의 응답을 절반씩 나눠 그 안에서만 따로 학습·평가한다.
- 기본형과 적대 학습형은 체크포인트·입력 길이·헤드·학습 예산·seed를 모두 같게 둔다.
- test는 --final로만 열며, 기본형/적대형 각각 한 번만 평가한다.
"""
import argparse
import copy
import json
import os
import platform
import random
import sys
import time
import warnings
from pathlib import Path

# 저장소 최상위 폴더를 import 경로에 추가한다 (evaluate.py, features/ 를 불러오기 위해)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")  # 화면 없이 그림 파일만 저장
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import transformers
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

import evaluate as ev                       # C의 공통 평가 코드
from features.build import load_splits      # 분할 CSV 로드·검사 (B)


# =========================================================
# 1. 모델 구성 요소
# =========================================================

class GradReverse(torch.autograd.Function):
    """기울기 반전 층: 순전파는 그대로, 역전파는 기울기에 -λ를 곱한다."""

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class AuthorTopicModel(nn.Module):
    """DeBERTa 인코더 + 작성자 헤드 + (GRL을 거친) 주제 헤드."""

    def __init__(self, backbone, n_author, n_topic, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        h = backbone.config.hidden_size
        self.drop = nn.Dropout(dropout)
        # 작성자 헤드: 선형층 하나 (기본형·적대형 동일)
        self.author_head = nn.Linear(h, n_author)
        # 주제 헤드: 은닉층 하나짜리 MLP. 너무 약하면 인코더가 쉽게 속여서 적대 신호가 약해진다.
        self.topic_head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, n_topic))

    def encode(self, input_ids, attention_mask):
        """마지막 층 토큰 벡터를 패딩을 뺀 평균으로 묶어 문서 벡터 하나를 만든다."""
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(out.dtype)
        return (out * m).sum(1) / m.sum(1).clamp(min=1.0)

    def forward(self, input_ids, attention_mask, lam=0.0):
        z = self.encode(input_ids, attention_mask)
        author_logits = self.author_head(self.drop(z))
        topic_logits = self.topic_head(GradReverse.apply(z, lam))
        return author_logits, topic_logits


# 모델·토크나이저 불러오기 (동작 확인 시 작은 모델로 바꿔 끼울 수 있도록 함수로 분리)
def load_tokenizer(name):
    return AutoTokenizer.from_pretrained(name)


def load_backbone(name):
    # 최신 transformers는 체크포인트에 저장된 정밀도(fp16)로 불러올 수 있다.
    # 헤드는 fp32이므로 인코더도 fp32로 맞춘다. GPU에서는 autocast가 계산만 fp16으로 바꾼다.
    return AutoModel.from_pretrained(name).float()


# =========================================================
# 2. 데이터 준비
# =========================================================

def set_seed(seed):
    """재현을 위해 난수 seed를 모두 고정한다 (GPU 연산 특성상 완전히 같지는 않을 수 있음)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def author_labels(df):
    """모델 이름 → 0~17 (evaluate.CLASSES 순서 = 알파벳순)."""
    return df.model.map({m: i for i, m in enumerate(ev.CLASSES)}).to_numpy()


def tokenize(tok, texts, max_len):
    """글을 토큰 ID로 바꾼다. 패딩은 배치마다 따로 한다 (동적 패딩으로 계산 절약)."""
    enc = tok(list(texts), truncation=True, max_length=max_len)
    n_trunc = sum(len(x) >= max_len for x in enc["input_ids"])  # 잘렸을 수 있는 글 수
    return enc["input_ids"], n_trunc


def batches(ids, idx, batch_size, pad_id, device, extra=()):
    """idx 순서대로 batch_size개씩 묶어 패딩한 텐서를 돌려준다. extra는 함께 묶을 라벨 배열들."""
    for s in range(0, len(idx), batch_size):
        b = idx[s:s + batch_size]
        L = max(len(ids[i]) for i in b)
        x = torch.full((len(b), L), pad_id, dtype=torch.long)
        m = torch.zeros((len(b), L), dtype=torch.long)
        for r, i in enumerate(b):
            x[r, :len(ids[i])] = torch.tensor(ids[i])
            m[r, :len(ids[i])] = 1
        yield (x.to(device), m.to(device)) + tuple(torch.as_tensor(e[b]).to(device) for e in extra)


# =========================================================
# 3. 학습·예측
# =========================================================

@torch.no_grad()
def predict(model, ids, batch_size, pad_id, device, amp):
    """작성자 확률 (n, 18)과 문서 벡터 z (n, hidden)를 돌려준다."""
    model.eval()
    P, Z = [], []
    idx = np.arange(len(ids))
    for x, m in batches(ids, idx, batch_size, pad_id, device):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            z = model.encode(x, m)
            logits = model.author_head(z)
        P.append(torch.softmax(logits.float(), -1).cpu().numpy())
        Z.append(z.float().cpu().numpy())
    return np.concatenate(P), np.concatenate(Z)


def train_one(args, lam, seed, data, device):
    """λ 하나 × seed 하나로 학습한다. 에폭마다 val BA를 재고, 가장 좋은 에폭의 가중치를 남긴다."""
    set_seed(seed)
    tok = data["tok"]
    model = AuthorTopicModel(load_backbone(args.model), 18, data["n_topic"], args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 인코더는 작은 학습률, 새로 붙인 헤드는 큰 학습률
    head_params = list(model.author_head.parameters()) + list(model.topic_head.parameters())
    head_ids = {id(p) for p in head_params}
    enc_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr},
                             {"params": head_params, "lr": args.head_lr}], weight_decay=0.01)
    n_train = len(data["ids"]["train"])
    steps = args.epochs * int(np.ceil(n_train / args.batch_size))
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp)
    ce = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history, best, best_state, bad = [], -1.0, None, 0
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = {"loss": 0.0, "author": 0.0, "topic": 0.0, "topic_hit": 0, "n": 0}
        order = rng.permutation(n_train)
        for x, m, ya, yt in batches(data["ids"]["train"], order, args.batch_size, tok.pad_token_id,
                                    device, extra=(data["y"]["train"], data["topic"])):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp):
                la, lt = model(x, m, lam)
                loss_a, loss_t = ce(la.float(), ya), ce(lt.float(), yt)
                # 두 손실을 더한다. 인코더에 대한 주제 손실의 방향·크기(-λ)는 GRL이 정한다.
                loss = loss_a + loss_t
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            k = len(ya)
            tot["loss"] += loss.item() * k
            tot["author"] += loss_a.item() * k
            tot["topic"] += loss_t.item() * k
            tot["topic_hit"] += (lt.argmax(1) == yt).sum().item()
            tot["n"] += k

        # 에폭 끝: val로 작성자 BA 측정 (에폭 선택에만 사용)
        P_val, _ = predict(model, data["ids"]["val"], args.batch_size * 2, tok.pad_token_id, device, args.amp)
        s = ev.score(data["y"]["val"], P_val)
        n = tot["n"]
        history.append({"lambda": lam, "seed": seed, "epoch": epoch,
                        "train_loss": tot["loss"] / n, "train_author_loss": tot["author"] / n,
                        "train_topic_loss": tot["topic"] / n,
                        # 학습 중 배치에서 주제 헤드가 맞힌 비율 (참고값; 진단은 아래 probe로 따로 함)
                        "train_topic_head_acc": tot["topic_hit"] / n,
                        "val_BA": s["model_BA"], "val_F1": s["model_F1"], "val_logloss": s["model_LogLoss"],
                        "elapsed_s": time.perf_counter() - t_start})
        print(f"λ={lam:<5} seed{seed} epoch{epoch}  loss={tot['loss']/n:.3f}  "
              f"author={tot['author']/n:.3f}  topic={tot['topic']/n:.3f}  "
              f"val_BA={s['model_BA']:.4f}  ({time.perf_counter()-t_start:.0f}s)", flush=True)

        # 조기 종료: val BA가 오르면 가중치 저장, patience번 연속 안 오르면 멈춤
        if s["model_BA"] > best:
            best, bad = s["model_BA"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  조기 종료: {args.patience}에폭 연속 val BA 개선 없음")
                break
    fit_s = time.perf_counter() - t_start
    peak_mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else float("nan")
    model.load_state_dict(best_state)
    info = {"best_epoch": best_epoch, "fit_seconds": fit_s, "peak_gpu_mem_gb": peak_mem,
            "n_params": n_params, "n_trainable_params": n_train_params}
    return model, history, info


# =========================================================
# 4. 주제 정보 진단 (probe)
# =========================================================

def probe(Z, prompt, author, seed):
    """문서 벡터 Z에 주제·작성자 정보가 얼마나 남았는지 간단한 선형 분류기로 잰다.

    같은 질문의 응답을 절반씩 나눠, 한쪽으로 로지스틱 회귀를 학습하고 다른 쪽으로 평가한다.
    두 쪽 모두 같은 질문 집합이라 주제(질문 ID)를 맞히는 문제가 성립한다.
    주제 정확도가 낮을수록 표현이 주제에 덜 의존한다는 뜻이다.
    """
    rng = np.random.default_rng(seed)
    side = np.zeros(len(Z), dtype=bool)
    for p in np.unique(prompt):
        i = np.flatnonzero(prompt == p)
        side[rng.permutation(i)[: len(i) // 2]] = True
    sc = StandardScaler().fit(Z[side])
    A, B = sc.transform(Z[side]), sc.transform(Z[~side])
    res = {}
    for name, y in [("topic", prompt), ("author", author)]:
        with warnings.catch_warnings():  # 수렴 경고는 진단 결과에 영향이 작아 숨긴다
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(A, y[side])
        pred = clf.predict(B)
        res[f"{name}_acc"] = accuracy_score(y[~side], pred)
        res[f"{name}_BA"] = balanced_accuracy_score(y[~side], pred)
    res["n_topics"] = len(np.unique(prompt))
    return res


# =========================================================
# 5. 실행 기록·그림
# =========================================================

def env_info(device):
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
            "cuda": torch.version.cuda, "device": str(device)}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def git_commit():
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def run_name(lam, seed):
    return f"A_deberta_lam{lam:g}_seed{seed}"


def plot(hist, summary, out):
    """왼쪽: 에폭별 val BA, 오른쪽: λ별 작성자 BA와 주제 probe 정확도."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for (lam, seed), h in hist.groupby(["lambda", "seed"]):
        axes[0].plot(h.epoch, h.val_BA, marker="o", label=f"λ={lam:g} seed{seed}")
    axes[0].axhline(1 / 18, color="gray", ls=":", lw=1, label="uniform 1/18")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("val model BA")
    axes[0].set_title("Validation BA per epoch")
    axes[0].legend(fontsize=8)
    g = summary.groupby("lambda").mean(numeric_only=True)
    x = np.arange(len(g))
    axes[1].bar(x - 0.2, g.model_BA, 0.4, label="author BA (val)")
    axes[1].bar(x + 0.2, g.probe_train_topic_acc, 0.4, label="topic probe acc (train halves)")
    axes[1].set_xticks(x, [f"λ={v:g}" for v in g.index])
    axes[1].set_title("Author vs topic information")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# =========================================================
# 6. 실행
# =========================================================

def prepare(args, include_test):
    frames = load_splits(args.data_dir, include_test=include_test)
    if args.debug_rows:  # 동작 확인용: 일부 행만 사용 (실제 실험에서는 쓰지 않음)
        frames = {k: v.groupby("model", group_keys=False).head(max(1, args.debug_rows // 18))
                  for k, v in frames.items()}
    tok = load_tokenizer(args.model)
    data = {"frames": frames, "tok": tok, "ids": {}, "y": {}, "n_trunc": {}}
    for k, df in frames.items():
        data["ids"][k], data["n_trunc"][k] = tokenize(tok, df.text, args.max_len)
        data["y"][k] = author_labels(df)
    # 주제 대용 라벨: train의 prompt_id를 0..(질문 수-1) 번호로 바꾼다 (val/test에는 쓰지 않음)
    topics = sorted(frames["train"].prompt_id.unique())
    data["topic"] = frames["train"].prompt_id.map({p: i for i, p in enumerate(topics)}).to_numpy()
    data["n_topic"] = len(topics)
    return data


def compare_val(args, device):
    data = prepare(args, include_test=False)
    print(f"토큰 길이 상한 {args.max_len}에 닿은 글 수: {data['n_trunc']}")
    commit = args.commit or git_commit()
    rows, hists = [], []
    for lam in args.lambdas:
        for seed in args.seeds:
            model, hist, info = train_one(args, lam, seed, data, device)
            hists += hist
            name = run_name(lam, seed)
            # 가장 좋은 에폭 가중치로 val 예측 저장 (C의 save_pred 형식)
            P_val, Z_val = predict(model, data["ids"]["val"], args.batch_size * 2,
                                   data["tok"].pad_token_id, device, args.amp)
            ev.save_pred(data["frames"]["val"].id, P_val, ev.CLASSES, args.pred_dir / f"val_{name}.csv")
            s = ev.score(data["y"]["val"], P_val)
            # 주제 정보 진단: train 절반 나누기(주 진단), val 절반 나누기(처음 보는 질문, 참고)
            _, Z_tr = predict(model, data["ids"]["train"], args.batch_size * 2,
                              data["tok"].pad_token_id, device, args.amp)
            fr = data["frames"]
            pt = probe(Z_tr, fr["train"].prompt_id.to_numpy(), data["y"]["train"], seed)
            pv = probe(Z_val, fr["val"].prompt_id.to_numpy(), data["y"]["val"], seed)
            rows.append({"run_id": name, "owner": "A", "model": "DeBERTa-v3-base",
                         "lambda": lam, "variant": "base" if lam == 0 else "topic_adversarial",
                         "feature_set": "raw text", "preprocessing": f"tokenizer max_len={args.max_len}",
                         "params": json.dumps({"lr": args.lr, "head_lr": args.head_lr, "epochs": args.epochs,
                                               "batch_size": args.batch_size, "patience": args.patience,
                                               "dropout": args.dropout, "amp": args.amp}),
                         "seed": seed, "split": "val", "n_samples": len(P_val),
                         "model_BA": s["model_BA"], "accuracy": s["model_Acc"], "macro_f1": s["model_F1"],
                         "log_loss": s["model_LogLoss"], "sibling_acc": s["sibling_Acc"],
                         "company_BA": s["company_BA"], "company_method": "prob_sum",
                         "company_BA_argmax": s["company_BA_argmax"],
                         "probe_train_topic_acc": pt["topic_acc"], "probe_train_author_BA": pt["author_BA"],
                         "probe_val_topic_acc": pv["topic_acc"], "probe_val_author_BA": pv["author_BA"],
                         **info, "code_commit": commit})
            if args.save_ckpt:  # test 평가에 쓰기 위해 가장 좋은 에폭 가중치 저장
                args.ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), args.ckpt_dir / f"{name}.pt")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary, hist = pd.DataFrame(rows), pd.DataFrame(hists)
    summary.to_csv(args.out_dir / "val_results.csv", index=False)
    hist.to_csv(args.out_dir / "history.csv", index=False)
    plot(hist, summary, args.out_dir / "fig_lambda_comparison.png")
    meta = {"model": args.model, "lambdas": args.lambdas, "seeds": args.seeds,
            "n_topics_train": data["n_topic"], "truncated_at_max_len": data["n_trunc"],
            "args": {k: str(v) for k, v in vars(args).items()},
            "environment": env_info(device), "code_commit": commit, "test_opened": False,
            "notes": ["주제 라벨은 train prompt_id 대용이며 실제 의미 주제와 다름",
                      "probe_train: train 응답을 질문별 절반으로 나눠 로지스틱 회귀로 측정 (주 진단)",
                      "probe_val: val 응답 안에서만 같은 방식으로 측정 (참고, train 주제 분류기 미사용)",
                      "우연 수준: 주제 train 1/105, val 1/22, 작성자 1/18"]}
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    cols = ["run_id", "best_epoch", "model_BA", "macro_f1", "company_BA", "sibling_acc",
            "probe_train_topic_acc", "probe_val_topic_acc", "fit_seconds", "peak_gpu_mem_gb"]
    pd.set_option("display.width", 200)
    print(summary[cols].round(4).to_string(index=False))
    print(f"저장 위치: {args.out_dir}")


def final_test(args, device):
    """저장된 체크포인트로 val·test를 예측한다. test 결과 파일이 이미 있으면 멈춘다."""
    target = args.out_dir / "test_results.csv"
    if target.exists() and not args.force:
        raise SystemExit(f"{target} 파일이 이미 있습니다: test는 이미 한 번 평가했습니다. test로 재튜닝하지 마세요.")
    data = prepare(args, include_test=True)
    commit = args.commit or git_commit()
    rows = []
    for lam in args.lambdas:
        for seed in args.seeds:
            name = run_name(lam, seed)
            ck = args.ckpt_dir / f"{name}.pt"
            if not ck.exists():
                raise SystemExit(f"{ck} 체크포인트가 없습니다. 먼저 val 비교를 실행하세요.")
            model = AuthorTopicModel(load_backbone(args.model), 18, data["n_topic"], args.dropout).to(device)
            model.load_state_dict(torch.load(ck, map_location=device))
            for split in ["val", "test"]:
                P, _ = predict(model, data["ids"][split], args.batch_size * 2,
                               data["tok"].pad_token_id, device, args.amp)
                ev.save_pred(data["frames"][split].id, P, ev.CLASSES, args.pred_dir / f"{split}_{name}.csv")
                s = ev.score(data["y"][split], P)
                rows.append({"run_id": name, "lambda": lam, "seed": seed, "split": split,
                             "n_samples": len(P), "model_BA": s["model_BA"], "accuracy": s["model_Acc"],
                             "macro_f1": s["model_F1"], "company_BA": s["company_BA"],
                             "company_method": "prob_sum", "sibling_acc": s["sibling_Acc"],
                             "code_commit": commit})
    res = pd.DataFrame(rows)
    res.to_csv(target, index=False)
    print(res.round(4).to_string(index=False))
    if args.debug_rows:  # 일부 행만 쓴 동작 확인에서는 신뢰구간을 계산하지 않음
        return
    # C의 공통 코드로 질문 단위 bootstrap 95% 신뢰구간 계산 (λ별)
    d = ev.load_data(*(args.data_dir / f"{s}.csv" for s in ["train", "val", "test"]))
    cis = []
    for lam in args.lambdas:
        run = ev.load_run([str(args.pred_dir / f"test_{run_name(lam, s)}.csv") for s in args.seeds],
                          d, split="test", open_test=True)
        cis.append(ev.bootstrap_ci(run).assign(**{"lambda": lam}))
    pd.concat(cis).to_csv(args.out_dir / "test_ci.csv")
    print(pd.concat(cis))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="microsoft/deberta-v3-base", help="Hugging Face 체크포인트 이름")
    p.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.1],
                   help="적대 강도 λ 목록 (0 = 기본형)")
    p.add_argument("--seeds", type=int, nargs="+", default=[42], help="seed 목록 (시간이 되면 42 43 44)")
    p.add_argument("--epochs", type=int, default=5, help="최대 에폭 수 (조기 종료 가능)")
    p.add_argument("--patience", type=int, default=2, help="val BA가 이 에폭 수만큼 안 오르면 멈춤")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5, help="인코더 학습률")
    p.add_argument("--head-lr", type=float, default=1e-3, help="분류 헤드 학습률")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-len", type=int, default=512, help="토큰 길이 상한 (train 최대 274단어라 사실상 잘림 없음)")
    p.add_argument("--no-amp", dest="amp", action="store_false", help="fp16 혼합 정밀도 끄기")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/split")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results/course/A/deberta")
    p.add_argument("--pred-dir", type=Path, default=ROOT / "preds")
    p.add_argument("--ckpt-dir", type=Path, default=ROOT / "artifacts/ckpt/A_deberta",
                   help="체크포인트 저장 폴더 (용량이 커서 GitHub에 올리지 않음)")
    p.add_argument("--no-save-ckpt", dest="save_ckpt", action="store_false")
    p.add_argument("--final", action="store_true", help="저장된 체크포인트로 test를 한 번 평가")
    p.add_argument("--force", action="store_true", help="이미 있는 test 결과를 덮어쓴다 (사용 자제)")
    p.add_argument("--commit", help=".git이 없을 때 기록할 코드 버전")
    p.add_argument("--debug-rows", type=int, default=0, help="동작 확인용으로 일부 행만 사용")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.pred_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        args.amp = False
        print("경고: GPU가 없어 CPU로 실행합니다. 실제 실험은 GPU에서 하세요.")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    if args.final:
        final_test(args, device)
    else:
        compare_val(args, device)


if __name__ == "__main__":
    main()
