"""D: Logistic Regression (D-1, D-2). train 학습, val 비교, test는 --final 로만."""
import argparse
import json
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import FeatureUnion, Pipeline
from scipy import sparse
from sklearn.preprocessing import StandardScaler

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
CS = [0.1, 1.0, 10.0, 100.0]
MAX_FEATURES = [1000, 5000, 30000]
CENTERING = ["raw", "centered", "concat"]
CENTER_DIM_CAP = 20000
BASE_MF = 30000
KINDS = ["style", "tfidf", "combined"]
MAX_ITER = 5000
PRIMARY_KIND = "style"

PREPROC = {
    "style_raw": "문체 32개, 스케일링 없음",
    "style": "문체 32개 -> StandardScaler(중심화), train fit",
    "tfidf": "문자 3~5-gram TF-IDF, train fit",
    "combined": "문체(중심화) + 문자 TF-IDF, train fit",
}


def make_tfidf(max_features):
    return TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=False,
                           min_df=2, max_features=max_features, sublinear_tf=True,
                           strip_accents=None, dtype=np.float64)


def make_feat(kind, max_features=BASE_MF):
    if kind == "style_raw":
        return Pipeline([("raw", StylometricTransformer())])
    style = Pipeline([("raw", StylometricTransformer()), ("scale", StandardScaler())])
    if kind == "style":
        return style
    if kind == "tfidf":
        return make_tfidf(max_features)
    if kind == "combined":
        return FeatureUnion([("style", style), ("tfidf", make_tfidf(max_features))])
    raise ValueError(kind)


def make_labels(df):
    return df.model.map({m: i for i, m in enumerate(ev.CLASSES)}).to_numpy()


def cfg_name(kind, C, mf, mode="raw"):
    base = f"{kind}_C{C}" if kind.startswith("style") else f"{kind}_mf{mf}_C{C}"
    return base if mode == "raw" else f"{base}_{mode}"


def center_by_prompt(X, pid):
    """같은 질문 응답들의 평균을 뺀다. 질문 주제 성분을 제거하기 위함."""
    X = np.asarray(X.todense() if sparse.issparse(X) else X, dtype=np.float64)
    out = X.copy()
    for pr in np.unique(pid):
        m = pid == pr
        out[m] -= X[m].mean(0)
    return out


def build_matrices(kind, mf, mode, frames, split):
    feat = make_feat(kind, mf)
    Xtr = feat.fit_transform(frames["train"].text)
    Xs = feat.transform(frames[split].text)
    if mode == "raw":
        return Xtr, Xs
    ptr = frames["train"].prompt_id.to_numpy()
    ps = frames[split].prompt_id.to_numpy()
    Ctr, Cs = center_by_prompt(Xtr, ptr), center_by_prompt(Xs, ps)
    if mode == "centered":
        return Ctr, Cs
    dense = lambda M: np.asarray(M.todense() if sparse.issparse(M) else M, dtype=np.float64)
    return np.hstack([dense(Xtr), Ctr]), np.hstack([dense(Xs), Cs])


def run(kind, C, frames, y, seed, mf=BASE_MF, split="val", mode="raw"):
    ytr = y["train"]
    Xtr, Xs = build_matrices(kind, mf, mode, frames, split)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        lr = LogisticRegression(C=C, max_iter=MAX_ITER, random_state=seed)
        t0 = time.perf_counter()
        lr.fit(Xtr, ytr)
        fit_s = time.perf_counter() - t0
        n_iter = int(np.max(lr.n_iter_))
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)

    t0 = time.perf_counter()
    P = np.zeros((Xs.shape[0], 18))
    P[:, lr.classes_] = lr.predict_proba(Xs)
    pred_s = time.perf_counter() - t0
    s = ev.score(y[split], P)
    Ptr = np.zeros((Xtr.shape[0], 18))
    Ptr[:, lr.classes_] = lr.predict_proba(Xtr)
    train_ba = balanced_accuracy_score(ytr, Ptr.argmax(1))
    name = cfg_name(kind, C, mf, mode)
    dim = lr.coef_.shape[1]
    print(f"  {name:36s} dim {dim:<6} train {train_ba:.4f} val {s['model_BA']:.4f}"
          f"  {'' if converged else '미수렴!'} {fit_s:5.1f}s", flush=True)

    row = {
        "run_id": f"D_lr_{name}_seed{seed}", "owner": "D", "model": "LogisticRegression",
        "config": name, "feature_set": kind, "centering": mode,
        "max_features": mf if kind in ("tfidf", "combined") else None,
        "preprocessing": PREPROC[kind] + ("" if mode == "raw" else f" + 질문 {mode}"),
        "params": json.dumps({"C": C, "max_features": mf, "centering": mode,
                              "max_iter": MAX_ITER, "solver": "lbfgs", "penalty": "l2"}),
        "seed": seed, "split": split, "n_samples": len(frames[split]), "n_features": dim,
        "samples_per_feature": round(len(frames["train"]) / dim, 3),
        "train_model_BA": train_ba, "company_method": "prob_sum",
        "accuracy": s["model_Acc"], "macro_f1": s["model_F1"],
        "log_loss": s["model_LogLoss"], "ece": s["model_ECE"],
        "sibling_acc": s["sibling_Acc"],
        "n_iter": n_iter, "converged": converged,
        "fit_seconds": round(fit_s, 2), "predict_seconds": round(pred_s, 3),
    }
    row.update(s)
    return row, lr, P


def run_dummy(frames, y):
    rows = []
    for strategy in ["most_frequent", "stratified", "prior"]:
        d = DummyClassifier(strategy=strategy, random_state=42).fit(frames["train"].text, y["train"])
        P = np.zeros((len(frames["val"]), 18))
        P[:, d.classes_] = d.predict_proba(frames["val"].text)
        s = ev.score(y["val"], P)
        rows.append({"run_id": f"D_dummy_{strategy}", "owner": "D", "model": "DummyClassifier",
                     "config": strategy, "feature_set": "-", "preprocessing": "-",
                     "params": json.dumps({"strategy": strategy}), "seed": 42, "split": "val",
                     "n_samples": len(frames["val"]),
                     "accuracy": s["model_Acc"], "macro_f1": s["model_F1"],
                     "company_method": "prob_sum", **s})
    return rows


def as_run(meta, y, P):
    return {"split": "val", "meta": meta, "y": y, "P": P[None]}


def ci_table(meta, y, Ps):
    recs = []
    for name, P in Ps.items():
        c = ev.bootstrap_ci(as_run(meta, y, P))
        recs.append({"config": name,
                     "model_BA_ci_low": round(c.loc["model_BA", "ci_low"], 4),
                     "model_BA_ci_high": round(c.loc["model_BA", "ci_high"], 4),
                     "company_BA_ci_low": round(c.loc["company_BA", "ci_low"], 4),
                     "company_BA_ci_high": round(c.loc["company_BA", "ci_high"], 4)})
    return pd.DataFrame(recs)


def pairwise_table(meta, y, Ps, best):
    base = as_run(meta, y, Ps[best])
    recs = []
    for name, P in Ps.items():
        if name == best:
            continue
        c = ev.compare(as_run(meta, y, P), base).loc["model_BA"]
        recs.append({"baseline": name, "vs": best,
                     "diff_mean": round(c.diff_mean, 4),
                     "ci_low": round(c.ci_low, 4), "ci_high": round(c.ci_high, 4),
                     "best_is_better": bool(c.improved),
                     "best_is_worse": bool(c.worse)})
    return pd.DataFrame(recs).sort_values("diff_mean", ascending=False)


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
    """확신이 높은 순으로 오분류 n개."""
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


def coefficient_table(lr, top_k=10):
    names = list(StylometricTransformer().get_feature_names_out())
    coef = lr.coef_
    recs = []
    for i, cls_idx in enumerate(lr.classes_):
        for j in np.argsort(-np.abs(coef[i]))[:top_k]:
            recs.append({"model": ev.CLASSES[cls_idx], "feature": names[j],
                         "coef": round(float(coef[i, j]), 4),
                         "abs_coef": round(float(abs(coef[i, j])), 4)})
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


def fig_grid(res, out):
    lr = res[res.model == "LogisticRegression"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    d1 = lr[lr.feature_set == PRIMARY_KIND].copy()
    d1["C"] = d1.params.map(lambda p: json.loads(p)["C"])
    d1 = d1.sort_values("C")
    axes[0].plot(d1.C.astype(str), d1.train_model_BA, "s--", label="train BA")
    axes[0].plot(d1.C.astype(str), d1.model_BA, "o-", label="val BA")
    axes[0].set(xlabel="C", ylabel="BA",
                title=T(f"D-1: {PRIMARY_KIND}, 규제", f"D-1: {PRIMARY_KIND}, regularisation"))
    axes[0].legend()
    axes[0].grid(alpha=.3)

    base = lr[(lr.max_features == BASE_MF) | (lr.feature_set == PRIMARY_KIND)]
    d2 = (base.sort_values("model_BA").drop_duplicates("feature_set", keep="last")
          .set_index("feature_set").reindex(KINDS).reset_index())
    x = np.arange(len(d2))
    axes[1].bar(x - .2, d2.train_model_BA, .4, label="train BA")
    axes[1].bar(x + .2, d2.model_BA, .4, label="val BA")
    axes[1].set_xticks(x, d2.feature_set)
    axes[1].set(ylabel="BA", title=T(f"D-2: 입력별 (mf={BASE_MF})",
                                     f"D-2: feature sets (mf={BASE_MF})"))
    axes[1].legend()
    axes[1].grid(alpha=.3, axis="y")

    sw = lr[lr.max_features.notna()]
    for kind in ["tfidf", "combined"]:
        g = (sw[sw.feature_set == kind].sort_values("model_BA")
             .drop_duplicates("max_features", keep="last").sort_values("max_features"))
        axes[2].plot(g.max_features.astype(int).astype(str), g.model_BA, "o-", label=f"{kind} val")
        axes[2].plot(g.max_features.astype(int).astype(str), g.train_model_BA, "s--",
                     alpha=.5, label=f"{kind} train")
    axes[2].set(xlabel="max_features", ylabel="BA",
                title=T("D-2c: 차원 수 (각 차원의 최고 C)",
                        "D-2c: dimension (best C each)"))
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(out / "fig_config_comparison.png", dpi=150)
    plt.close(fig)


def compare_val(args, frames, y, out, commit):
    rows = run_dummy(frames, y)
    Ps, pipes = {}, {}
    meta = frames["val"][["id", "prompt_id"]]

    print("\n[D-1] 문체 32개, C 비교")
    for C in CS:
        row, pipe, P = run(PRIMARY_KIND, C, frames, y, 42)
        rows.append(row)
        Ps[row["config"]] = P
        pipes[row["config"]] = pipe

    d1 = pd.DataFrame([r for r in rows if r.get("feature_set") == PRIMARY_KIND])
    C_fixed = json.loads(d1.loc[d1.model_BA.idxmax(), "params"])["C"]
    print(f"-> 선택 C = {C_fixed}")

    print("\n[D-2 보조] 스케일링 없는 문체")
    rows.append(run("style_raw", C_fixed, frames, y, 42)[0])

    print("\n[D-2 · D-2b · D-2c] 입력 x 차원 x C")
    for kind in ["tfidf", "combined"]:
        for mf in MAX_FEATURES:
            for C in CS:
                row, pipe, P = run(kind, C, frames, y, 42, mf)
                rows.append(row)
                Ps[row["config"]] = P

    raw_lr = pd.DataFrame([r for r in rows if r["model"] == "LogisticRegression"])
    raw_best = raw_lr.loc[raw_lr.model_BA.idxmax()]
    best_mf = json.loads(raw_best.params)["max_features"]
    print(f"\n[D-2d] 질문 중심화 (원본 최고: {raw_best.config})")
    targets = [(PRIMARY_KIND, None)] + ([(raw_best.feature_set, best_mf)]
                                        if raw_best.feature_set != PRIMARY_KIND else [])
    for kind, mf in targets:
        mf = mf or BASE_MF
        for mode in ["centered", "concat"]:
            dim = (32 if kind == "style" else mf + 32) * (2 if mode == "concat" else 1)
            if dim > CENTER_DIM_CAP:
                print(f"  건너뜀 {kind} mf={mf} {mode}: {dim}차원 > 상한 {CENTER_DIM_CAP}")
                continue
            for C in CS:
                row, pipe, P = run(kind, C, frames, y, 42, mf, mode=mode)
                rows.append(row)
                Ps[row["config"]] = P

    lr = pd.DataFrame([r for r in rows if r["model"] == "LogisticRegression"])
    best = lr.loc[lr.model_BA.idxmax()]
    print(f"\n-> 최고 설정: {best.config} (val model_BA {best.model_BA:.4f})")

    print(f"\n[seed 변동] {best.config}")
    bp = json.loads(best.params)
    for seed in SEEDS:
        if seed == 42:
            P = Ps[best.config]
        else:
            row, _, P = run(best.feature_set, bp["C"], frames, y, seed,
                            bp["max_features"], mode=bp.get("centering", "raw"))
            rows.append(row)
        ev.save_pred(frames["val"].id, P, ev.CLASSES,
                     args.pred_dir / f"val_D_lr_{best.config}_seed{seed}.csv")

    res = pd.DataFrame(rows).assign(code_commit=commit)
    res.to_csv(out / "val_results.csv", index=False)
    (res[res.model == "LogisticRegression"]
     .groupby("config")[["train_model_BA", "model_BA", "accuracy", "macro_f1",
                         "log_loss", "company_BA", "fit_seconds"]]
     .agg(["mean", "std"]).round(4).to_csv(out / "val_summary.csv"))

    print("\n[bootstrap] prompt 단위 재표집")
    ci_table(meta, y["val"], Ps).to_csv(out / "val_ci.csv", index=False)
    pw = pairwise_table(meta, y["val"], Ps, best.config)
    pw.to_csv(out / "val_pairwise.csv", index=False)
    n_sig = int(pw.best_is_better.sum())
    print(f"  {best.config} 가 통계적으로 앞선 설정: {n_sig}/{len(pw)}")

    P_best = Ps[best.config]
    error_table(frames, y, P_best).to_csv(out / "val_errors_seed42.csv", index=False)
    calibration_table(y["val"], P_best).to_csv(out / "val_calibration.csv", index=False)
    per_class_table(y["val"], P_best).to_csv(out / "val_per_class.csv", index=False)
    sibling_table(y["val"], P_best).to_csv(out / "val_sibling_errors.csv", index=False)
    company_table(y["val"], P_best).to_csv(
        out / "val_company_breakdown.csv", index=False)
    coefficient_table(pipes[cfg_name(PRIMARY_KIND, C_fixed, BASE_MF)]).to_csv(
        out / "coefficients_style.csv", index=False)
    fig_confusion(y["val"], P_best, out, best.config)
    fig_grid(res, out)

    cols = ["run_id", "feature_set", "n_features", "samples_per_feature", "train_model_BA",
            "model_BA", "accuracy", "macro_f1", "log_loss", "company_BA",
            "converged", "fit_seconds"]
    print("\n=== val 결과 ===")
    print(res[[c for c in cols if c in res.columns]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return res, {"selected_C": C_fixed, "best_config": best.config}


def final_test(args, frames, y, out, commit, name):
    target = out / f"test_results_{name}.csv"
    if target.exists() and not args.force:
        sys.exit(f"{target} 가 이미 있다. 재평가는 튜닝이 된다. 덮어쓰려면 --force.")
    parts = name.split("_")
    kind = parts[0]
    mf = int([p[2:] for p in parts if p.startswith("mf")][0]) if "mf" in name else BASE_MF
    C = float([p[1:] for p in parts if p.startswith("C")][0])
    rows = []
    for seed in SEEDS:
        row, pipe, P = run(kind, C, frames, y, seed, mf, split="test")
        rows.append(row)
        ev.save_pred(frames["test"].id, P, ev.CLASSES,
                     args.pred_dir / f"test_D_lr_{name}_seed{seed}.csv")
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
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/split")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results/course/D/logistic_regression")
    p.add_argument("--pred-dir", type=Path, default=ROOT / "preds")
    p.add_argument("--final", help="test를 1회 평가할 설정 이름 (예: combined_mf5000_C100.0)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--commit")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.pred_dir.mkdir(parents=True, exist_ok=True)
    commit = args.commit or git_commit()

    env = {"python": platform.python_version(), "platform": platform.platform(),
           "numpy": np.__version__, "pandas": pd.__version__,
           "scikit-learn": sklearn.__version__, "scipy": scipy.__version__,
           "code_commit": commit}
    print("환경:", json.dumps(env, ensure_ascii=False))

    open_test = args.final is not None
    frames = load_splits(args.data_dir, include_test=open_test)
    y = {k: make_labels(df) for k, df in frames.items()}
    print("분할 크기:", {k: len(v) for k, v in frames.items()})

    meta = {"env": env, "seeds": SEEDS, "Cs": CS, "kinds": KINDS,
            "max_features": MAX_FEATURES, "centering": CENTERING, "max_iter": MAX_ITER}
    if open_test:
        final_test(args, frames, y, args.out_dir, commit, args.final)
        meta["final_config"] = args.final
    else:
        meta.update(compare_val(args, frames, y, args.out_dir, commit)[1])

    (args.out_dir / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n산출물: {args.out_dir}")


if __name__ == "__main__":
    main()
