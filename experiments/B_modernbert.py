"""B: ModernBERT-base classifier with/without supervised contrastive loss (B-5, B-6).

GPU (Colab) only. Run from repo root:
    python -m experiments.B_modernbert --lam 0      # B-5 baseline: CE only
    python -m experiments.B_modernbert --lam 0.1    # B-6: CE + lam * SupCon
Both variants share model, max_len, sampler, optimizer, epochs and seed; only lam differs.
Train on train, pick the epoch by val model BA. Test is never opened here.
"""
import argparse
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from sklearn.metrics import silhouette_score
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from evaluate import CLASSES, save_pred, score
from features.build import load_splits

CKPT = "answerdotai/ModernBERT-base"


def supcon(z, y, t):
    """Supervised contrastive loss (Khosla et al. 2020); positives = same author model in the batch."""
    z = F.normalize(z.float(), dim=1)
    eye = torch.eye(len(y), dtype=torch.bool, device=z.device)
    sim = (z @ z.T / t).masked_fill(eye, -1e9)
    pos = (y[:, None] == y[None]) & ~eye
    logp = sim - sim.logsumexp(1, keepdim=True)
    has = pos.any(1)
    return -((logp * pos).sum(1)[has] / pos.sum(1)[has]).mean()


def pk_batches(y, p, k, rng):
    """Each batch: p classes x k samples, so every sample has >= 1 positive.
    In this split each prompt is answered once per model, so same-class pairs come from different prompts."""
    queues = {c: list(rng.permutation(np.where(y == c)[0])) for c in np.unique(y)}
    out = []
    while True:
        ready = [c for c, q in queues.items() if len(q) >= k]
        if len(ready) < p:
            return out
        batch = []
        for c in rng.choice(ready, p, replace=False):
            batch += [queues[c].pop() for _ in range(k)]
        out.append(batch)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(CKPT)
        self.head = torch.nn.Linear(self.enc.config.hidden_size, 18)

    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        emb = (h * m).sum(1) / m.sum(1)  # mean pooling
        return self.head(emb), emb


@torch.no_grad()
def predict(net, tok, texts, args, dev):
    net.eval()
    logits, embs = [], []
    for i in range(0, len(texts), 64):
        b = tok(list(texts[i:i + 64]), truncation=True, max_length=args.max_len, padding=True, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            lg, e = net(b.input_ids, b.attention_mask)
        logits.append(lg.float().cpu()); embs.append(e.float().cpu())
    return torch.cat(logits).softmax(1).numpy(), torch.cat(embs).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--p", type=int, default=8, help="classes per batch")
    ap.add_argument("--k", type=int, default=2, help="samples per class per batch")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "GPU runtime required (Colab: Runtime > Change runtime type > T4 GPU)"
    dev = "cuda"
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    frames = load_splits("data/split")
    tr, va = frames["train"], frames["val"].sort_values("id").reset_index(drop=True)
    idx = {m: i for i, m in enumerate(CLASSES)}
    ytr, yva = tr.model.map(idx).to_numpy(), va.model.map(idx).to_numpy()
    name = f"B_modernbert_lam{args.lam:g}"
    out = Path("results/course/B/modernbert") / f"{name}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(CKPT)
    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(pk_batches(ytr, args.p, args.k, np.random.default_rng(0))) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.cuda.amp.GradScaler()
    torch.cuda.reset_peak_memory_stats()

    log, best, t_start = [], None, time.perf_counter()
    for ep in range(1, args.epochs + 1):
        net.train()
        ce_sum = con_sum = 0.0
        batches = pk_batches(ytr, args.p, args.k, rng)
        for b in batches:
            enc = tok(list(tr.text.iloc[b]), truncation=True, max_length=args.max_len, padding=True,
                      return_tensors="pt").to(dev)
            y = torch.tensor(ytr[b], device=dev)
            with torch.autocast("cuda", dtype=torch.float16):
                lg, emb = net(enc.input_ids, enc.attention_mask)
            ce = F.cross_entropy(lg.float(), y)
            con = supcon(emb, y, args.temp)
            loss = ce + args.lam * con  # lam=0 -> plain classifier; con is logged either way
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            ce_sum += ce.item(); con_sum += con.item()
        P, E = predict(net, tok, va.text.to_numpy(), args, dev)
        s = score(yva, P)
        sil = float(silhouette_score(E, yva, metric="cosine"))
        row = {"epoch": ep, "train_ce": ce_sum / len(batches), "train_supcon": con_sum / len(batches),
               "val_model_BA": s["model_BA"], "val_acc": s["model_Acc"], "val_macro_f1": s["model_F1"],
               "val_company_BA": s["company_BA"], "val_sibling_Acc": s["sibling_Acc"],
               "val_LogLoss": s["model_LogLoss"], "val_silhouette_cos": sil,
               "elapsed_s": round(time.perf_counter() - t_start, 1)}
        log.append(row); print(json.dumps(row))
        if best is None or s["model_BA"] > best[0]["model_BA"]:
            best = (s, P, ep, sil)
    fit_s = time.perf_counter() - t_start

    s, P, ep, sil = best
    save_pred(va.id, P, CLASSES, Path("preds") / f"val_{name}_seed{args.seed}.csv")
    pd.DataFrame(log).to_csv(out / "epoch_log.csv", index=False)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    summary = {"run_id": name, "owner": "B", "model": CKPT, "feature_set": "raw text",
               "preprocessing": f"tokenizer truncation max_len={args.max_len}, mean pooling",
               "params": json.dumps({k: v for k, v in vars(args).items() if k != "seed"}), "seed": args.seed,
               "split": "val", "n_samples": len(yva), "model_BA": s["model_BA"], "accuracy": s["model_Acc"],
               "macro_f1": s["model_F1"], "company_BA": s["company_BA"], "company_method": "prob_sum(softmax)",
               "fit_seconds": round(fit_s, 1), "predict_seconds": None, "code_commit": commit,
               "best_epoch": ep, "val_silhouette_cos": sil,
               "trainable_params": sum(p.numel() for p in net.parameters() if p.requires_grad),
               "peak_gpu_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
               "gpu": torch.cuda.get_device_name(0),
               "versions": {"python": platform.python_version(), "torch": torch.__version__,
                            "transformers": transformers.__version__},
               "selection": "epoch with best val model BA", "test_opened": False,
               **{f"cal_{k}": v for k, v in s.items()}}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    # self-check: SupCon is lower when same-class embeddings coincide than when they are random
    y0 = torch.tensor([0, 0, 1, 1])
    assert supcon(torch.tensor([[1., 0], [1, 0], [0, 1], [0, 1]]), y0, 0.1) < supcon(torch.randn(4, 2), y0, 0.1)
    assert all(len(set(np.array([0, 0, 1, 1, 2, 2])[b])) == 2 for b in pk_batches(np.array([0, 0, 1, 1, 2, 2]), 2, 2, np.random.default_rng(0)))
    main()
