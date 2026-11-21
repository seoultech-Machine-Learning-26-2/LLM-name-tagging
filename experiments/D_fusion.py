"""D: Sentence-BERT 문체 융합 (D-3, D-4). 임베딩 동결, test는 --final 로만."""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluate as ev                                    # noqa: E402
from features.build import load_splits                   # noqa: E402
from features.stylometry import StylometricTransformer   # noqa: E402


def setup_font():
    for path in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                 "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
                 "C:/Windows/Fonts/malgun.ttf"]:
        if Path(path).exists():
            fm.fontManager.addfont(path)
            plt.rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


KO = setup_font()
T = (lambda ko, en: ko if KO else en)

SEEDS = [42, 43, 44]
BASE_BRANCHES = ["semantic", "style", "fusion"]
CENTERING = ["raw", "centered"]
BRANCHES = [b if m == "raw" else f"{b}_centered"
            for b in BASE_BRANCHES for m in CENTERING]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HP = {"hidden": 256, "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-4,
      "batch_size": 32, "epochs": 40, "patience": 8}


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, dropout, n_class=18):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_class),
        )

    def forward(self, x):
        return self.net(x)


def make_labels(df):
    return df.model.map({m: i for i, m in enumerate(ev.CLASSES)}).to_numpy()


def encode(frames, encoder_name, cache_dir, batch_size=64):
    from sentence_transformers import SentenceTransformer
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = encoder_name.replace("/", "_")
    out, model = {}, None
    for split, df in frames.items():
        path = cache_dir / f"{tag}_{split}.npy"
        if path.exists():
            out[split] = np.load(path)
            print(f"  캐시 사용 {split} {out[split].shape}", flush=True)
            continue
        if model is None:
            model = SentenceTransformer(encoder_name, device=DEVICE)
            model.eval()
        t0 = time.perf_counter()
        with torch.no_grad():
            E = model.encode(df.text.tolist(), batch_size=batch_size,
                             convert_to_numpy=True, show_progress_bar=False)
        np.save(path, E)
        out[split] = E
        print(f"  인코딩 {split} {E.shape} {time.perf_counter()-t0:.1f}s", flush=True)
    if model is not None:
        del model
        torch.cuda.empty_cache()
    return out


def center_by_prompt(X, pid):
    X = np.asarray(X, dtype=np.float64)
    out = X.copy()
    for pr in np.unique(pid):
        m = pid == pr
        out[m] -= X[m].mean(0)
    return out


def build_inputs(frames, emb):
    style_pipe = Pipeline([("raw", StylometricTransformer()), ("scale", StandardScaler())])
    style = {"train": style_pipe.fit_transform(frames["train"].text)}
    for s in frames:
        if s != "train":
            style[s] = style_pipe.transform(frames[s].text)

    emb_scaler = StandardScaler().fit(emb["train"])
    sem = {s: emb_scaler.transform(emb[s]) for s in emb}

    base = {"semantic": sem, "style": style,
            "fusion": {s: np.hstack([sem[s], style[s]]) for s in emb}}
    pid = {s: frames[s].prompt_id.to_numpy() for s in frames}
    X = {}
    for b, d in base.items():
        X[b] = d
        X[f"{b}_centered"] = {s: center_by_prompt(v, pid[s]) for s, v in d.items()}
    return {b: {s: np.asarray(v, dtype=np.float32) for s, v in d.items()} for b, d in X.items()}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_branch(Xb, y, seed, hp, log=None):
    set_seed(seed)
    Xtr = torch.tensor(Xb["train"])
    ytr = torch.tensor(y["train"], dtype=torch.long)
    Xva = torch.tensor(Xb["val"]).to(DEVICE)

    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=hp["batch_size"],
                        shuffle=True, generator=g, drop_last=True)

    model = MLP(Xtr.shape[1], hp["hidden"], hp["dropout"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    lossf = nn.CrossEntropyLoss()

    best = {"ba": -1.0, "epoch": -1, "state": None}
    bad = 0
    t0 = time.perf_counter()
    for epoch in range(1, hp["epochs"] + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xva).argmax(1).cpu().numpy()
        ba = balanced_accuracy_score(y["val"], pred)
        if log is not None:
            log.append({"seed": seed, "epoch": epoch, "val_BA": round(float(ba), 4)})
        if ba > best["ba"]:
            best = {"ba": ba, "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    model.load_state_dict(best["state"])
    return model, best["epoch"], time.perf_counter() - t0


def predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X).to(DEVICE))
        return torch.softmax(logits, 1).cpu().numpy()


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_branch(branch, X, y, seed, hp, encoder, split="val", curve=None):
    model, best_epoch, fit_s = train_branch(X[branch], y, seed, hp, log=curve)
    t0 = time.perf_counter()
    P = predict_proba(model, X[branch][split])
    pred_s = time.perf_counter() - t0
    s = ev.score(y[split], P)
    train_ba = balanced_accuracy_score(y["train"],
                                       predict_proba(model, X[branch]["train"]).argmax(1))
    print(f"  {branch:9s} seed {seed}  best_epoch {best_epoch:3d}  "
          f"val_BA {s['model_BA']:.4f}  {fit_s:5.1f}s", flush=True)
    row = {
        "run_id": f"D_fusion_{branch}_seed{seed}", "owner": "D", "model": "SBERT+MLP",
        "config": branch, "feature_set": branch.replace("_centered", ""),
        "centering": "centered" if branch.endswith("_centered") else "raw",
        "encoder": encoder,
        "preprocessing": "임베딩 동결 + StandardScaler(train fit)"
                         + (" + 질문 중심화" if branch.endswith("_centered") else ""),
        "params": json.dumps(hp), "seed": seed, "split": split,
        "n_samples": len(y[split]), "n_features": X[branch][split].shape[1],
        "n_trainable_params": n_params(model), "best_epoch": best_epoch,
        "train_model_BA": train_ba, "company_method": "prob_sum",
        "accuracy": s["model_Acc"], "macro_f1": s["model_F1"],
        "log_loss": s["model_LogLoss"], "ece": s["model_ECE"],
        "sibling_acc": s["sibling_Acc"],
        "fit_seconds": round(fit_s, 2), "predict_seconds": round(pred_s, 3),
    }
    row.update(s)
    return row, model, P


def calibration_table(y_true, P, bins=5):
    conf = P.max(1)
    ok = P.argmax(1) == y_true
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    recs = []
    for b in range(bins):
        m = idx == b
        if m.any():
            recs.append({"bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}", "n": int(m.sum()),
                         "mean_confidence": round(float(conf[m].mean()), 4),
                         "actual_accuracy": round(float(ok[m].mean()), 4),
                         "gap": round(float(conf[m].mean() - ok[m].mean()), 4)})
    return pd.DataFrame(recs)


def per_class_table(y_true, P):
    pred = P.argmax(1)
    recs = []
    for i, name in enumerate(ev.CLASSES):
        m = y_true == i
        recs.append({"model": name, "n_val": int(m.sum()),
                     "recall": round(float((pred[m] == i).mean()), 4) if m.any() else np.nan,
                     "n_predicted": int((pred == i).sum()),
                     "precision": round(float((y_true[pred == i] == i).mean()), 4)
                     if (pred == i).any() else np.nan})
    return pd.DataFrame(recs).sort_values("recall")


def company_table(y_true, P):
    pred = P.argmax(1)
    predc = ev.to_company(P).argmax(1)
    yc = ev.M2C[y_true]
    recs = []
    for c in range(6):
        m = yc == c
        k = int((ev.M2C == c).sum())
        sm = m & (ev.M2C[pred] == c) & ev.MULTI[yc]
        recs.append({"company": ev.COMPANIES[c], "n_models": k, "n_val": int(m.sum()),
                     "company_recall": round(float((predc[m] == c).mean()), 4),
                     "company_chance": round(1 / 6, 4),
                     "within_company_n": int(sm.sum()),
                     "within_company_acc": round(float((pred[sm] == y_true[sm]).mean()), 4)
                     if sm.any() else np.nan,
                     "within_company_chance": round(1 / k, 4) if k > 1 else np.nan})
    df = pd.DataFrame(recs).sort_values("company_recall", ascending=False)
    multi = df[df.n_models > 1]
    df = pd.concat([df, pd.DataFrame([
        {"company": "ALL(6)", "company_recall": round(df.company_recall.mean(), 4),
         "company_chance": round(1 / 6, 4)},
        {"company": "MULTI_ONLY(4)", "company_recall": round(multi.company_recall.mean(), 4),
         "company_chance": round(1 / len(multi), 4),
         "within_company_acc": round(
             float(np.average(multi.within_company_acc, weights=multi.within_company_n)), 4)},
    ])], ignore_index=True)
    return df


def sibling_table(y_true, P):
    pred = P.argmax(1)
    w = pred != y_true
    same = ev.M2C[y_true[w]] == ev.M2C[pred[w]]
    return pd.DataFrame([{"n_wrong": int(w.sum()),
                          "same_company_errors": int(same.sum()),
                          "same_company_rate": round(float(same.mean()), 4),
                          "chance_rate": round(1 / 6, 4)}])


def error_table(frames, y, P, n=5):
    pred = P.argmax(1)
    wrong = np.flatnonzero(pred != y["val"])
    picked = wrong[np.argsort(-P.max(1)[wrong])][:n]
    val = frames["val"].reset_index(drop=True)
    recs = []
    for i in picked:
        recs.append({
            "id": val.id[i], "prompt_id": val.prompt_id[i],
            "true_model": ev.CLASSES[y["val"][i]], "pred_model": ev.CLASSES[pred[i]],
            "pred_prob": round(float(P[i].max()), 4),
            "true_prob": round(float(P[i, y["val"][i]]), 4),
            "same_company": bool(ev.M2C[y["val"][i]] == ev.M2C[pred[i]]),
            "text_head": str(val.text[i])[:300].replace("\n", " "),
        })
    return pd.DataFrame(recs)


def fig_confusion(y_true, P, out, name):
    M = confusion_matrix(y_true, P.argmax(1), labels=range(18), normalize="true")
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(18), ev.CLASSES, rotation=90, fontsize=8)
    ax.set_yticks(range(18), ev.CLASSES, fontsize=8)
    ax.set(xlabel=T("예측", "predicted"), ylabel=T("실제", "true"),
           title=T(f"val 혼동행렬 (행 정규화) — {name}",
                   f"val confusion (row-normalised) — {name}"))
    fig.colorbar(im, ax=ax, fraction=.046)
    fig.tight_layout()
    fig.savefig(out / "fig_confusion_val.png", dpi=150)
    plt.close(fig)


def fig_branches(res, curves, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    g = res.groupby("config")[["train_model_BA", "model_BA"]].agg(["mean", "std"])
    g = g.reindex(BRANCHES)
    x = np.arange(len(g))
    axes[0].bar(x - .2, g[("train_model_BA", "mean")], .4, label="train BA")
    axes[0].bar(x + .2, g[("model_BA", "mean")], .4,
                yerr=g[("model_BA", "std")].fillna(0), capsize=4, label="val BA")
    axes[0].set_xticks(x, BRANCHES, rotation=20, ha="right", fontsize=8)
    axes[0].set(ylabel="BA", title=T("D-4: 분기별 (seed 3개 평균)",
                                     "D-4: branches (mean of 3 seeds)"))
    axes[0].legend()
    axes[0].grid(alpha=.3, axis="y")

    for b in BRANCHES:
        c = curves[curves.config == b]
        if c.empty:
            continue
        m = c.groupby("epoch").val_BA.mean()
        axes[1].plot(m.index, m.values, label=b)
    axes[1].set(xlabel="epoch", ylabel="val BA",
                title=T("학습 곡선 (seed 3개 평균)", "val BA by epoch (mean of 3 seeds)"))
    axes[1].legend()
    axes[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "fig_branch_comparison.png", dpi=150)
    plt.close(fig)


def compare_val(args, frames, y, X, out, commit):
    rows, curves, Ps = [], [], {}
    for branch in BRANCHES:
        print(f"\n[{branch}] 입력 {X[branch]['train'].shape[1]}차원")
        for seed in SEEDS:
            curve = []
            row, model, P = run_branch(branch, X, y, seed, HP, args.encoder, curve=curve)
            rows.append(row)
            curves += [dict(c, config=branch) for c in curve]
            ev.save_pred(frames["val"].id, P, ev.CLASSES,
                         args.pred_dir / f"val_D_fusion_{branch}_seed{seed}.csv")
            if seed == 42:
                Ps[branch] = P

    res = pd.DataFrame(rows).assign(code_commit=commit)
    res.to_csv(out / "val_results.csv", index=False)
    curves = pd.DataFrame(curves)
    curves.to_csv(out / "val_learning_curve.csv", index=False)

    summary = (res.groupby("config")[["train_model_BA", "model_BA", "accuracy", "macro_f1",
                                      "log_loss", "ece", "sibling_acc", "company_BA",
                                      "n_trainable_params", "best_epoch", "fit_seconds"]]
               .agg(["mean", "std"]).round(4).reindex(BRANCHES))
    summary.to_csv(out / "val_summary.csv")

    best = res.groupby("config").model_BA.mean().idxmax()
    best_P = Ps[best]

    calibration_table(y["val"], best_P).to_csv(out / "val_calibration.csv", index=False)
    per_class_table(y["val"], best_P).to_csv(out / "val_per_class.csv", index=False)
    sibling_table(y["val"], best_P).to_csv(out / "val_sibling_errors.csv", index=False)
    company_table(y["val"], best_P).to_csv(
        out / "val_company_breakdown.csv", index=False)
    error_table(frames, y, best_P).to_csv(out / "val_errors_seed42.csv", index=False)
    fig_confusion(y["val"], best_P, out, best)
    fig_branches(res, curves, out)

    print("\n=== 분기별 요약 (seed 3개) ===")
    print(summary[[("train_model_BA", "mean"), ("model_BA", "mean"), ("model_BA", "std"),
                   ("macro_f1", "mean"), ("company_BA", "mean"),
                   ("n_trainable_params", "mean")]]
          .to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\n-> 최고 분기: {best}")
    return res, {"best_branch": best}


def final_test(args, frames, y, X, out, commit, branch):
    target = out / f"test_results_{branch}.csv"
    if target.exists() and not args.force:
        sys.exit(f"{target} 가 이미 있다. 재평가는 튜닝이 된다. 덮어쓰려면 --force.")
    rows = []
    for seed in SEEDS:
        row, model, P = run_branch(branch, X, y, seed, HP, args.encoder, split="test")
        rows.append(row)
        ev.save_pred(frames["test"].id, P, ev.CLASSES,
                     args.pred_dir / f"test_D_fusion_{branch}_seed{seed}.csv")
    res = pd.DataFrame(rows).assign(code_commit=commit)
    res.to_csv(target, index=False)
    print(res[["run_id", "model_BA", "accuracy", "macro_f1", "company_BA"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return res


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/split")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results/course/D/fusion")
    p.add_argument("--pred-dir", type=Path, default=ROOT / "preds")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "results/course/D/embeddings")
    p.add_argument("--final", choices=BRANCHES, help="test를 1회 평가할 분기")
    p.add_argument("--force", action="store_true")
    p.add_argument("--epochs", type=int)
    p.add_argument("--commit")
    args = p.parse_args()
    if args.epochs:
        HP["epochs"] = args.epochs
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.pred_dir.mkdir(parents=True, exist_ok=True)
    commit = args.commit or git_commit()

    env = {"python": platform.python_version(), "platform": platform.platform(),
           "numpy": np.__version__, "pandas": pd.__version__,
           "scikit-learn": sklearn.__version__, "torch": torch.__version__,
           "device": DEVICE,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "encoder": args.encoder, "code_commit": commit}
    print("환경:", json.dumps(env, ensure_ascii=False))

    open_test = args.final is not None
    frames = load_splits(args.data_dir, include_test=open_test)
    y = {k: make_labels(df) for k, df in frames.items()}
    print("분할 크기:", {k: len(v) for k, v in frames.items()})

    print("\n[임베딩] 동결 Sentence-BERT")
    emb = encode(frames, args.encoder, args.cache_dir)
    X = build_inputs(frames, emb)
    print("입력 차원:", {b: X[b]["train"].shape[1] for b in BRANCHES})

    meta = {"env": env, "seeds": SEEDS, "branches": BRANCHES, "centering": CENTERING, "hp": HP,
            "input_dims": {b: int(X[b]["train"].shape[1]) for b in BRANCHES}}
    if open_test:
        final_test(args, frames, y, X, args.out_dir, commit, args.final)
        meta["final_branch"] = args.final
    else:
        meta.update(compare_val(args, frames, y, X, args.out_dir, commit)[1])

    (args.out_dir / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n산출물: {args.out_dir}")


if __name__ == "__main__":
    main()
