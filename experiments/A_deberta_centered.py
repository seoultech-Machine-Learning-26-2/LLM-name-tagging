"""[A] DeBERTa-v3-base + 질문 중심화 (2026-11-18 회의 Q-10 결정 2, 적용 규칙 Q-11).

사용법 (저장소 최상위 폴더에서, GPU가 켜진 Kaggle 권장):
    python experiments/A_deberta_centered.py

조건 (seed 42, 최대 10에폭, 나머지 설정은 A_deberta.py 10에폭 실행과 같음):
    qb_nocenter_lam0   ③ 질문 배치, 중심화 없음, λ=0   — 배치 방식만 바꾼 대조군
    qb_center_lam0     ① 질문 배치 + 중심화, λ=0       — 중심화 효과
    qb_center_lam0.1   ② 질문 배치 + 중심화, λ=0.1     — 중심화 + 주제 적대 학습

- 질문 배치: 배치 하나 = 같은 prompt_id의 응답 전부(18개, train의 한 질문만 17개).
- 질문 중심화: 문서 벡터 z에서 같은 배치(= 같은 질문)의 평균을 빼고 [z ; z − 질문 평균]을 작성자 헤드에 넣는다.
  평균은 각 분할 안에서 같은 질문의 응답끼리만 계산하며 정답(model)은 쓰지 않는다.
- 주제 적대 학습(GRL)은 기존과 같이 중심화 전 z에 건다.
- val만 학습·평가한다. test는 읽지 않는다(예측은 12월에 다시 학습해서 만든다).
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup

import evaluate as ev
from features.build import load_splits
import A_deberta as base  # 기존 코드의 부품을 그대로 쓴다 (GRL, 토크나이저·인코더 로드, probe 등)

CONDITIONS = {  # 이름: (중심화 여부, λ)
    "qb_nocenter_lam0": (False, 0.0),
    "qb_center_lam0": (True, 0.0),
    "qb_center_lam0.1": (True, 0.1),
}


class CenteredAuthorTopicModel(base.AuthorTopicModel):
    """기존 모델과 같고, 중심화를 켜면 작성자 헤드 입력이 [z ; z − 질문 평균] (2 × hidden)이 된다."""

    def __init__(self, backbone, n_author, n_topic, center, dropout=0.1):
        super().__init__(backbone, n_author, n_topic, dropout)
        self.center = center
        if center:
            self.author_head = nn.Linear(2 * backbone.config.hidden_size, n_author)

    def author_input(self, z):
        """z: 한 질문의 응답들 (배치 = 질문 하나)."""
        if not self.center:
            return z
        return torch.cat([z, z - z.mean(0, keepdim=True)], dim=1)

    def forward(self, input_ids, attention_mask, lam=0.0):
        z = self.encode(input_ids, attention_mask)
        author_logits = self.author_head(self.drop(self.author_input(z)))
        topic_logits = self.topic_head(base.GradReverse.apply(z, lam))
        return author_logits, topic_logits, z


def question_groups(prompt_ids):
    """prompt_id별 행 번호 목록 (분할 안에서만)."""
    s = pd.Series(np.arange(len(prompt_ids)), index=np.asarray(prompt_ids))
    return [g.to_numpy() for _, g in s.groupby(level=0, sort=True)]


def pad_batch(ids, rows, pad_id, device):
    L = max(len(ids[i]) for i in rows)
    x = torch.full((len(rows), L), pad_id, dtype=torch.long)
    m = torch.zeros((len(rows), L), dtype=torch.long)
    for r, i in enumerate(rows):
        x[r, :len(ids[i])] = torch.tensor(ids[i])
        m[r, :len(ids[i])] = 1
    return x.to(device), m.to(device)


@torch.no_grad()
def predict(model, ids, groups, pad_id, device, amp):
    """질문 단위로 묶어 예측. 반환: 작성자 확률 (n, 18), 중심화 전 z, 작성자 헤드 입력 h."""
    model.eval()
    n = len(ids)
    P = np.zeros((n, 18))
    Z = H = None
    for rows in groups:
        x, m = pad_batch(ids, rows, pad_id, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            z = model.encode(x, m)
        z = z.float()
        h = model.author_input(z)
        P[rows] = torch.softmax(model.author_head(h).float(), -1).cpu().numpy()
        if Z is None:
            Z, H = np.zeros((n, z.shape[1])), np.zeros((n, h.shape[1]))
        Z[rows], H[rows] = z.cpu().numpy(), h.cpu().numpy()
    return P, Z, H


def train_one(args, cond, data, device, seed):
    center, lam = CONDITIONS[cond]
    base.set_seed(seed)
    tok = data["tok"]
    model = CenteredAuthorTopicModel(base.load_backbone(args.model), 18, data["n_topic"], center,
                                     args.dropout).to(device)
    head_params = list(model.author_head.parameters()) + list(model.topic_head.parameters())
    head_ids = {id(p) for p in head_params}
    enc_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr},
                             {"params": head_params, "lr": args.head_lr}], weight_decay=0.01)
    groups_tr = data["groups"]["train"]
    steps = args.epochs * len(groups_tr)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp)
    ce = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    y_tr = torch.as_tensor(data["y"]["train"])
    t_tr = torch.as_tensor(data["topic"])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history, best, best_state, best_epoch, bad = [], -1.0, None, 0, 0
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = {"loss": 0.0, "author": 0.0, "topic": 0.0, "topic_hit": 0, "n": 0}
        for gi in rng.permutation(len(groups_tr)):  # 질문 순서만 섞는다
            rows = groups_tr[gi]
            x, m = pad_batch(data["ids"]["train"], rows, tok.pad_token_id, device)
            ya, yt = y_tr[rows].to(device), t_tr[rows].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp):
                la, lt, _ = model(x, m, lam)
                loss_a, loss_t = ce(la.float(), ya), ce(lt.float(), yt)
                loss = loss_a + loss_t
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            k = len(rows)
            tot["loss"] += loss.item() * k
            tot["author"] += loss_a.item() * k
            tot["topic"] += loss_t.item() * k
            tot["topic_hit"] += (lt.argmax(1) == yt).sum().item()
            tot["n"] += k

        P_val, _, _ = predict(model, data["ids"]["val"], data["groups"]["val"], tok.pad_token_id, device, args.amp)
        s = ev.score(data["y"]["val"], P_val)
        n = tot["n"]
        history.append({"condition": cond, "center": center, "lambda": lam, "seed": seed, "epoch": epoch,
                        "train_loss": tot["loss"] / n, "train_author_loss": tot["author"] / n,
                        "train_topic_loss": tot["topic"] / n, "train_topic_head_acc": tot["topic_hit"] / n,
                        "val_BA": s["model_BA"], "val_F1": s["model_F1"], "val_logloss": s["model_LogLoss"],
                        "elapsed_s": time.perf_counter() - t_start})
        print(f"{cond:17s} seed{seed} epoch{epoch}  loss={tot['loss']/n:.3f}  author={tot['author']/n:.3f}  "
              f"topic={tot['topic']/n:.3f}  val_BA={s['model_BA']:.4f}  ({time.perf_counter()-t_start:.0f}s)",
              flush=True)
        if s["model_BA"] > best:
            best, bad, best_epoch = s["model_BA"], 0, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  조기 종료: {args.patience}에폭 연속 val BA 개선 없음")
                break
    fit_s = time.perf_counter() - t_start
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else float("nan")
    model.load_state_dict(best_state)
    info = {"best_epoch": best_epoch, "fit_seconds": fit_s, "peak_gpu_mem_gb": peak,
            "n_params": sum(p.numel() for p in model.parameters()),
            "n_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    return model, history, info


def plot(hist, summary, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for cond, h in hist.groupby("condition", sort=False):
        axes[0].plot(h.epoch, h.val_BA, marker="o", label=cond)
    axes[0].axhline(1 / 18, color="gray", ls=":", lw=1, label="uniform 1/18")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("val model BA")
    axes[0].set_title("Validation BA per epoch (seed 42)")
    axes[0].legend(fontsize=8)
    x = np.arange(len(summary))
    axes[1].bar(x - 0.2, summary.model_BA, 0.4, label="author BA (val)")
    axes[1].bar(x + 0.2, summary.company_BA, 0.4, label="company BA (val)")
    axes[1].set_xticks(x, summary.condition, fontsize=8)
    axes[1].set_title("Best epoch, val")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="microsoft/deberta-v3-base")
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/split")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results/course/A/deberta_centered")
    p.add_argument("--pred-dir", type=Path, default=ROOT / "preds/deberta_centered")
    p.add_argument("--commit", help=".git이 없을 때 기록할 코드 버전")
    p.add_argument("--debug-rows", type=int, default=0, help="동작 확인용: train·val 질문을 앞에서부터 이 개수만 사용")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.pred_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        args.amp = False
        print("경고: GPU가 없어 CPU로 실행합니다. 실제 실험은 GPU에서 하세요.")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    frames = load_splits(args.data_dir)  # train·val만 (test는 읽지 않음)
    if args.debug_rows:
        frames = {k: v[v.prompt_id.isin(sorted(v.prompt_id.unique())[:args.debug_rows])] for k, v in frames.items()}
    frames = {k: v.reset_index(drop=True) for k, v in frames.items()}
    tok = base.load_tokenizer(args.model)
    data = {"tok": tok, "ids": {}, "y": {}, "groups": {}, "n_trunc": {}}
    for k, df in frames.items():
        data["ids"][k], data["n_trunc"][k] = base.tokenize(tok, df.text, args.max_len)
        data["y"][k] = base.author_labels(df)
        data["groups"][k] = question_groups(df.prompt_id)
    topics = sorted(frames["train"].prompt_id.unique())
    data["topic"] = frames["train"].prompt_id.map({t: i for i, t in enumerate(topics)}).to_numpy()
    data["n_topic"] = len(topics)
    print(f"토큰 길이 상한 {args.max_len}에 닿은 글 수: {data['n_trunc']}")
    commit = args.commit or base.git_commit()

    rows, hists = [], []
    for cond in args.conditions:
        for seed in args.seeds:
            center, lam = CONDITIONS[cond]
            model, hist, info = train_one(args, cond, data, device, seed)
            hists += hist
            name = f"A_deberta_{cond}_seed{seed}"
            P, Z, H = predict(model, data["ids"]["val"], data["groups"]["val"], tok.pad_token_id, device, args.amp)
            ev.save_pred(frames["val"].id, P, ev.CLASSES, args.pred_dir / f"val_{name}.csv")
            s = ev.score(data["y"]["val"], P)
            _, Z_tr, H_tr = predict(model, data["ids"]["train"], data["groups"]["train"], tok.pad_token_id,
                                    device, args.amp)
            pt_z = base.probe(Z_tr, frames["train"].prompt_id.to_numpy(), data["y"]["train"], seed)
            pv_z = base.probe(Z, frames["val"].prompt_id.to_numpy(), data["y"]["val"], seed)
            pv_h = base.probe(H, frames["val"].prompt_id.to_numpy(), data["y"]["val"], seed)
            rows.append({"run_id": name, "owner": "A", "model": "DeBERTa-v3-base", "condition": cond,
                         "question_batch": True, "center": center, "lambda": lam,
                         "question_grouping": "질문 묶음 이용" if center else "응답 단독",
                         "feature_set": "raw text", "preprocessing": f"tokenizer max_len={args.max_len}",
                         "params": json.dumps({"lr": args.lr, "head_lr": args.head_lr, "epochs": args.epochs,
                                               "patience": args.patience, "batch": "question (17~18)",
                                               "dropout": args.dropout, "amp": args.amp}),
                         "seed": seed, "split": "val", "n_samples": len(P),
                         "model_BA": s["model_BA"], "accuracy": s["model_Acc"], "macro_f1": s["model_F1"],
                         "log_loss": s["model_LogLoss"], "ece": s["model_ECE"], "sibling_acc": s["sibling_Acc"],
                         "company_BA": s["company_BA"], "company_method": "prob_sum",
                         "company_BA_argmax": s["company_BA_argmax"],
                         "probe_train_topic_acc_z": pt_z["topic_acc"], "probe_val_topic_acc_z": pv_z["topic_acc"],
                         "probe_val_topic_acc_head_input": pv_h["topic_acc"],
                         "probe_val_author_BA_head_input": pv_h["author_BA"],
                         **info, "code_commit": commit})
            print(f"  → {name}: best_epoch={info['best_epoch']}  val_BA={s['model_BA']:.4f}  "
                  f"company_BA={s['company_BA']:.4f}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary, hist = pd.DataFrame(rows), pd.DataFrame(hists)
    summary.to_csv(args.out_dir / "val_results.csv", index=False)
    hist.to_csv(args.out_dir / "history.csv", index=False)
    plot(hist, summary, args.out_dir / "fig_condition_comparison.png")
    meta = {"model": args.model, "conditions": {c: {"center": CONDITIONS[c][0], "lambda": CONDITIONS[c][1]}
                                                for c in args.conditions},
            "seeds": args.seeds, "n_topics_train": data["n_topic"], "truncated_at_max_len": data["n_trunc"],
            "args": {k: str(v) for k, v in vars(args).items()},
            "environment": base.env_info(device), "code_commit": commit, "test_read": False,
            "notes": ["배치 = 질문 하나의 응답 전부, 중심화 평균도 그 안에서만 계산 (정답 사용 안 함)",
                      "중심화 조건의 작성자 헤드 입력: [z ; z − 질문 평균]",
                      "probe_*_z: 중심화 전 z, probe_*_head_input: 작성자 헤드 입력 (중심화 조건은 [z ; z − 평균])",
                      "test는 읽지 않음. 비교 대상(무작위 배치 16, 중심화 없음)은 results/course/A/deberta_ep10"]}
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    cols = ["run_id", "best_epoch", "model_BA", "macro_f1", "company_BA", "sibling_acc", "log_loss", "ece",
            "probe_train_topic_acc_z", "fit_seconds", "peak_gpu_mem_gb"]
    pd.set_option("display.width", 220)
    print(summary[cols].round(4).to_string(index=False))
    print(f"저장 위치: {args.out_dir}, 예측: {args.pred_dir}")


if __name__ == "__main__":
    main()
