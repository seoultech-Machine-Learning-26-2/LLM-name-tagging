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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluate as ev                                    # noqa: E402  C의 공통 평가 코드
from features.build import load_splits, make_extractor   # noqa: E402  B의 공통 특징
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
CS = [0.1, 1.0, 10.0]
KINDS = ["style", "tfidf", "combined"]
MAX_ITER = 5000          # 미수렴은 converged 열에 기록한다
PRIMARY_KIND = "style"   # D-1에서 C를 고르는 기준 입력

PREPROC = {
    "style_raw": "문체 32개, 스케일링 없음",
    "style": "문체 32개 -> StandardScaler(중심화), train에만 fit",
    "tfidf": "문자 3~5-gram TF-IDF(max 30000), train에만 fit",
    "combined": "문체(중심화 스케일링) + 문자 TF-IDF 결합, train에만 fit",
}


def make_feat(kind):
    if kind == "style_raw":
        return Pipeline([("raw", StylometricTransformer())])
    style = Pipeline([("raw", StylometricTransformer()), ("scale", StandardScaler())])
    if kind == "style":
        return style
    if kind == "tfidf":
        return make_extractor("tfidf")
    if kind == "combined":
        return FeatureUnion([("style", style), ("tfidf", make_extractor("tfidf"))])
    raise ValueError(f"모르는 입력 종류: {kind}")


def make_pipe(kind, C, seed):
    return Pipeline([
        ("feat", make_feat(kind)),
        ("lr", LogisticRegression(C=C, max_iter=MAX_ITER, random_state=seed)),
    ])


def predict_proba18(pipe, X):
    P = np.zeros((len(X), 18))
    P[:, pipe.named_steps["lr"].classes_] = pipe.predict_proba(X)
    return P


def make_labels(df):
    return df.model.map({m: i for i, m in enumerate(ev.CLASSES)}).to_numpy()


def run(kind, C, frames, y, seed, split="val"):
    Xtr, ytr = frames["train"].text, y["train"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipe = make_pipe(kind, C, seed)
        t0 = time.perf_counter()
        pipe.fit(Xtr, ytr)
        fit_s = time.perf_counter() - t0
        n_iter = int(np.max(pipe.named_steps["lr"].n_iter_))
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    print(f"  {kind:9s} C={C:<5} fit {fit_s:6.1f}s  iter {n_iter:5d}"
          f"  {'수렴' if converged else '미수렴!'}", flush=True)

    t0 = time.perf_counter()
    P = predict_proba18(pipe, frames[split].text)
    pred_s = time.perf_counter() - t0
    s = ev.score(y[split], P)
    train_ba = balanced_accuracy_score(ytr, predict_proba18(pipe, Xtr).argmax(1))

    row = {
        "run_id": f"D_lr_{kind}_C{C}_seed{seed}", "owner": "D", "model": "LogisticRegression",
        "config": f"{kind}_C{C}", "feature_set": kind, "preprocessing": PREPROC[kind],
        "params": json.dumps({"C": C, "max_iter": MAX_ITER, "solver": "lbfgs", "penalty": "l2"}),
        "seed": seed, "split": split, "n_samples": len(frames[split]),
        "n_features": pipe.named_steps["lr"].coef_.shape[1],
        "train_model_BA": train_ba, "company_method": "prob_sum",
        "accuracy": s["model_Acc"], "macro_f1": s["model_F1"],
        "log_loss": s["model_LogLoss"], "ece": s["model_ECE"],
        "sibling_acc": s["sibling_Acc"],
        "n_iter": n_iter, "converged": converged,
        "fit_seconds": round(fit_s, 2), "predict_seconds": round(pred_s, 3),
    }
    row.update(s)
    return row, pipe, P


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
        true_m, pred_m = ev.CLASSES[y["val"][i]], ev.CLASSES[pred[i]]
        recs.append({
            "id": val.id[i], "prompt_id": val.prompt_id[i],
            "true_model": true_m, "pred_model": pred_m,
            "pred_prob": round(float(P[i].max()), 4),
            "true_prob": round(float(P[i, y["val"][i]]), 4),
            "same_company": bool(ev.M2C[y["val"][i]] == ev.M2C[pred[i]]),
            "text_head": str(val.text[i])[:300].replace("\n", " "),
        })
    return pd.DataFrame(recs)


def coefficient_table(pipe, top_k=10):
    names = list(StylometricTransformer().get_feature_names_out())
    coef = pipe.named_steps["lr"].coef_
    recs = []
    for i, cls_idx in enumerate(pipe.named_steps["lr"].classes_):
        for j in np.argsort(-np.abs(coef[i]))[:top_k]:
            recs.append({"model": ev.CLASSES[cls_idx], "feature": names[j],
                         "coef": round(float(coef[i, j]), 4),
                         "abs_coef": round(float(abs(coef[i, j])), 4)})
    return pd.DataFrame(recs)


def fig_confusion(y_true, P, out, name):
    from sklearn.metrics import confusion_matrix
    M = confusion_matrix(y_true, P.argmax(1), labels=range(18), normalize="true")
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(18), ev.CLASSES, rotation=90, fontsize=8)
    ax.set_yticks(range(18), ev.CLASSES, fontsize=8)
    ax.set_xlabel("예측")
    ax.set_ylabel("실제")
    ax.set_title(f"val 혼동행렬 (행 정규화) — {name}")
    fig.colorbar(im, ax=ax, fraction=.046)
    fig.tight_layout()
    path = out / "fig_confusion_val.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_comparison(res, out):
    d1 = res[(res.feature_set == PRIMARY_KIND) & (res.seed == 42) &
             res.run_id.str.contains("_C")].drop_duplicates("config").sort_values("params")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    d1 = d1.assign(C=d1.params.map(lambda p: json.loads(p)["C"])).sort_values("C")
    axes[0].plot(d1.C.astype(str), d1.train_model_BA, "s--", label="train BA")
    axes[0].plot(d1.C.astype(str), d1.model_BA, "o-", label="val BA")
    axes[0].set(xlabel="C (작을수록 규제 강함)", ylabel="BA",
                title=f"D-1: {PRIMARY_KIND} 입력, 규제 비교")
    axes[0].legend(); axes[0].grid(alpha=.3)

    d2 = res[(res.seed == 42) & res.config.str.startswith(tuple(KINDS))] \
        .sort_values("model_BA").drop_duplicates("feature_set", keep="last")
    d2 = d2.set_index("feature_set").reindex(KINDS).reset_index()
    x = np.arange(len(d2))
    axes[1].bar(x - .2, d2.train_model_BA, .4, label="train BA")
    axes[1].bar(x + .2, d2.model_BA, .4, label="val BA")
    axes[1].set_xticks(x, d2.feature_set)
    axes[1].set(ylabel="BA", title=T("D-2: 입력별 (각 입력의 최고 C)",
                                      "D-2: feature sets (best C each)"))
    axes[1].legend(); axes[1].grid(alpha=.3, axis="y")

    fig.tight_layout()
    path = out / "fig_config_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def compare_val(args, frames, y, out, commit):
    rows = run_dummy(frames, y)
    pipes, Ps = {}, {}

    print("\n[D-1] 문체 32개, C 비교")
    for C in CS:
        row, pipe, P = run(PRIMARY_KIND, C, frames, y, 42)
        rows.append(row); pipes[f"{PRIMARY_KIND}_C{C}"] = pipe; Ps[f"{PRIMARY_KIND}_C{C}"] = P

    d1 = pd.DataFrame(rows)
    d1 = d1[d1.feature_set == PRIMARY_KIND]
    C_fixed = json.loads(d1.loc[d1.model_BA.idxmax(), "params"])["C"]
    print(f"-> D-1 선택 C = {C_fixed} (val model_BA 최대)")

    print(f"\n[D-2] C={C_fixed} 고정, 입력 비교")
    for kind in KINDS:
        key = f"{kind}_C{C_fixed}"
        if key in pipes:
            continue
        row, pipe, P = run(kind, C_fixed, frames, y, 42)
        rows.append(row); pipes[key] = pipe; Ps[key] = P

    print("\n[D-2 보조] 스케일링 효과 (문체, 스케일링 없음)")
    row, pipe, P = run("style_raw", C_fixed, frames, y, 42)
    rows.append(row)

    print("\n[D-2b] 입력마다 C 재튜닝")
    for kind in KINDS:
        for C in CS:
            key = f"{kind}_C{C}"
            if key in pipes:
                continue
            row, pipe, P = run(kind, C, frames, y, 42)
            rows.append(row); pipes[key] = pipe; Ps[key] = P

    res = pd.DataFrame(rows)
    lr = res[res.model == "LogisticRegression"]
    best = lr.loc[lr.model_BA.idxmax()]
    best_key = best.config
    print(f"\n-> 최고 설정: {best_key} (val model_BA {best.model_BA:.4f})")

    print(f"\n[seed 변동] {best_key}")
    for seed in SEEDS:
        if seed == 42:
            continue
        kind, C = best.feature_set, json.loads(best.params)["C"]
        row, _, _ = run(kind, C, frames, y, seed)
        rows.append(row)

    res = pd.DataFrame(rows).assign(code_commit=commit)
    res.to_csv(out / "val_results.csv", index=False)

    summary = (res[res.model == "LogisticRegression"]
               .groupby("config")[["train_model_BA", "model_BA", "accuracy", "macro_f1",
                                   "log_loss", "company_BA", "fit_seconds"]]
               .agg(["mean", "std"]).round(4))
    summary.to_csv(out / "val_summary.csv")

    P_best = Ps[best_key]
    ev.save_pred(frames["val"].id, P_best, ev.CLASSES,
                 args.pred_dir / f"val_D_lr_{best_key}_seed42.csv")
    error_table(frames, y, P_best).to_csv(out / "val_errors_seed42.csv", index=False)
    calibration_table(y["val"], P_best).to_csv(out / "val_calibration.csv", index=False)
    per_class_table(y["val"], P_best).to_csv(out / "val_per_class.csv", index=False)
    sibling_table(y["val"], P_best).to_csv(out / "val_sibling_errors.csv", index=False)
    coefficient_table(pipes[f"{PRIMARY_KIND}_C{C_fixed}"]).to_csv(
        out / "coefficients_style.csv", index=False)
    fig_confusion(y["val"], P_best, out, best_key)
    fig_comparison(res, out)

    cols = ["run_id", "feature_set", "train_model_BA", "model_BA", "accuracy",
            "macro_f1", "log_loss", "company_BA", "converged", "fit_seconds"]
    print("\n=== val 결과 ===")
    print(res[[c for c in cols if c in res.columns]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return res, {"selected_C": C_fixed, "best_config": best_key}


def final_test(args, frames, y, out, commit, name):
    target = out / f"test_results_{name}.csv"
    if target.exists() and not args.force:
        sys.exit(f"{target} 가 이미 있다. test 재평가는 튜닝이 된다. 덮어쓰려면 --force.")
    kind, C = name.rsplit("_C", 1)
    rows = []
    for seed in SEEDS:
        row, pipe, P = run(kind, float(C), frames, y, seed, split="test")
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
                                       cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/split", help="A의 분할 CSV 폴더")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results/course/D/logistic_regression")
    p.add_argument("--pred-dir", type=Path, default=ROOT / "preds")
    p.add_argument("--final", help="test를 1회 평가할 설정 이름 (예: combined_C10.0)")
    p.add_argument("--force", action="store_true", help="기존 test 결과 덮어쓰기 (사용 자제)")
    p.add_argument("--commit", help=".git이 없을 때 기록할 코드 버전")
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

    meta = {"env": env, "seeds": SEEDS, "Cs": CS, "kinds": KINDS, "max_iter": MAX_ITER}
    if open_test:
        final_test(args, frames, y, args.out_dir, commit, args.final)
        meta["final_config"] = args.final
    else:
        _, chosen = compare_val(args, frames, y, args.out_dir, commit)
        meta.update(chosen)

    (args.out_dir / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n산출물: {args.out_dir}")


if __name__ == "__main__":
    main()
