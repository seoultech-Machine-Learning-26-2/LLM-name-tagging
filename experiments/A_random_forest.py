"""[A] 공통 문체 특징 32개로 학습하는 Random Forest 모델.

사용법 (저장소 최상위 폴더에서):
    python experiments/A_random_forest.py               # val로 설정 비교 (test는 열지 않음)
    python experiments/A_random_forest.py --final d10   # 설정 고정 후 test 한 번 평가

- 입력: A의 분할 + B의 문체 특징 32개 (트리 모델이라 스케일링 없이 원래 값 사용)
- 비교: 트리 수 50/100/300, 깊이 5/10/None. 한 번에 하나만 바꾸고 seed 42/43/44로 반복
- 선택: val BA 평균 1위와 seed 표준편차 이내인 설정 중 val log loss가 가장 낮은 것 (C의 규칙)
- 평가: C의 evaluate.py
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # evaluate.py, features/ 를 불러오기 위해

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

import evaluate as ev
from features.build import load_splits
from features.stylometry import FEATURE_NAMES, StylometricTransformer

SEEDS = [42, 43, 44]
BASE = dict(n_estimators=100, max_depth=None, min_samples_leaf=1)

# 이름: (바꾼 조건, 설정값). 기본값(n100)에서 하나씩만 바꾼다.
CONFIGS = {
    "n50":  ("n_estimators", {**BASE, "n_estimators": 50}),
    "n100": ("base",         dict(BASE)),
    "n300": ("n_estimators", {**BASE, "n_estimators": 300}),
    "d5":   ("max_depth",    {**BASE, "max_depth": 5}),
    "d10":  ("max_depth",    {**BASE, "max_depth": 10}),
}


# ---------------------------------------------------------------- 데이터

def make_features(frames):
    """각 분할의 text → 문체 특징 (n, 32). 글마다 고정 공식이라 학습되는 값이 없다."""
    st = StylometricTransformer().fit(frames["train"].text)
    return {k: np.asarray(st.transform(df.text), dtype=float) for k, df in frames.items()}


def make_labels(df):
    """모델 이름 → 0~17 (evaluate.CLASSES 순서)."""
    return df.model.map({m: i for i, m in enumerate(ev.CLASSES)}).to_numpy()


# ---------------------------------------------------------------- 학습·예측

def predict_proba18(model, X):
    """(n, 18) 확률 행렬. 열 순서는 evaluate.CLASSES."""
    P = np.zeros((len(X), 18))
    P[:, model.classes_] = model.predict_proba(X)
    return P


def fit_rf(params, seed, X_train, y_train, n_jobs):
    """RF를 train으로 학습하고 (모델, 학습 시간, train BA)를 돌려준다."""
    clf = RandomForestClassifier(**params, random_state=seed, n_jobs=n_jobs)
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0
    train_ba = balanced_accuracy_score(y_train, clf.predict(X_train))
    return clf, fit_s, train_ba


def timed_predict(model, X):
    t0 = time.perf_counter()
    P = predict_proba18(model, X)
    return P, time.perf_counter() - t0


# ---------------------------------------------------------------- 결과 기록

def result_row(name, cond, params, seed, split, s, fit_s, pred_s, train_ba, commit, n):
    """통합 결과표 한 줄 (계획 문서의 필수 열)."""
    return {
        "run_id": f"A_rf_{name}_seed{seed}", "owner": "A", "model": "RandomForest",
        "config": name, "changed": cond, "feature_set": "style32",
        "preprocessing": "raw (no scaling)", "params": json.dumps(params), "seed": seed,
        "split": split, "n_samples": n, "train_model_BA": train_ba,
        "model_BA": s["model_BA"], "accuracy": s["model_Acc"], "macro_f1": s["model_F1"],
        "log_loss": s["model_LogLoss"], "sibling_acc": s["sibling_Acc"],
        "company_BA": s["company_BA"], "company_method": "prob_sum",
        "company_BA_argmax": s["company_BA_argmax"],
        "fit_seconds": fit_s, "predict_seconds": pred_s, "code_commit": commit,
    }


def summarize(res):
    """설정별 seed 3개 평균·표준편차."""
    rf = res[res.model == "RandomForest"]
    summary = rf.groupby("config").agg(
        changed=("changed", "first"),
        train_BA_mean=("train_model_BA", "mean"), train_BA_std=("train_model_BA", "std"),
        val_BA_mean=("model_BA", "mean"), val_BA_std=("model_BA", "std"),
        val_acc_mean=("accuracy", "mean"), val_f1_mean=("macro_f1", "mean"),
        val_logloss_mean=("log_loss", "mean"), val_company_BA_mean=("company_BA", "mean"),
        val_sibling_mean=("sibling_acc", "mean"), fit_seconds_mean=("fit_seconds", "mean"),
    ).loc[list(CONFIGS)]
    summary["gap_train_minus_val"] = summary.train_BA_mean - summary.val_BA_mean  # 과적합 정도
    return summary


def select_config(summary):
    """(규칙으로 고른 설정, val BA 평균 1위 설정)."""
    best = summary.sort_values("val_BA_mean", ascending=False).iloc[0]
    close = summary[summary.val_BA_mean >= best.val_BA_mean - best.val_BA_std]
    return close.sort_values("val_logloss_mean").index[0], best.name


def feature_importance(models, X_val, y_val, X_train, n_jobs):
    """MDI(train 기준)와 순열 중요도(val에서 섞었을 때 BA 하락)를 seed 평균으로."""
    mdi = np.stack([m.feature_importances_ for m in models])
    perm = np.stack([
        permutation_importance(m, X_val, y_val, scoring="balanced_accuracy", n_repeats=10,
                               random_state=seed, n_jobs=n_jobs).importances_mean
        for m, seed in zip(models, SEEDS)])
    return pd.DataFrame({
        "feature": FEATURE_NAMES, "constant_in_train": X_train.std(0) == 0,
        "mdi_mean": mdi.mean(0), "mdi_std": mdi.std(0),
        "perm_mean": perm.mean(0), "perm_std": perm.std(0),
    })


def error_table(val_df, X_val, y_val, P):
    """val 오분류 목록 (확신이 높았던 오류부터)."""
    v = val_df.reset_index(drop=True)
    pred = P.argmax(1)
    err = pd.DataFrame({
        "id": v.id, "prompt_id": v.prompt_id, "true": v.model,
        "pred": [ev.CLASSES[i] for i in pred],
        "pred_prob": P.max(1), "true_prob": P[np.arange(len(P)), y_val],
        "same_company": ev.M2C[pred] == ev.M2C[y_val],
        "char_count": X_val[:, 0], "punct_per_100": X_val[:, 12],
        "em_dash_per_100": X_val[:, FEATURE_NAMES.index("em_dash_per_100_chars")],
        "text": v.text,
    })[pred != y_val]
    return err.sort_values("pred_prob", ascending=False)


# ---------------------------------------------------------------- 그림

def plot_comparison(summary, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    panels = [(axes[0], ["n50", "n100", "n300"], ["50", "100", "300"], "n_estimators (max_depth=None)"),
              (axes[1], ["d5", "d10", "n100"], ["5", "10", "None"], "max_depth (n_estimators=100)")]
    for ax, keys, xs, title in panels:
        s = summary.loc[keys]
        ax.errorbar(xs, s.train_BA_mean, yerr=s.train_BA_std, marker="o", capsize=3, label="train BA")
        ax.errorbar(xs, s.val_BA_mean, yerr=s.val_BA_std, marker="s", capsize=3, label="val BA")
        ax.axhline(1 / 18, color="gray", ls=":", lw=1, label="uniform 1/18")
        ax.set_title(title)
        ax.set_xlabel(title.split()[0])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("model BA (18 classes), mean ± std over seeds 42/43/44")
    axes[0].legend()
    save_fig(fig, path)


def plot_importance(imp, path):
    imp = imp.sort_values("mdi_mean").tail(20)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    axes[0].barh(imp.feature, imp.mdi_mean, xerr=imp.mdi_std)
    axes[0].set_title("MDI importance (train, mean over seeds)")
    axes[1].barh(imp.feature, imp.perm_mean, xerr=imp.perm_std, color="tab:orange")
    axes[1].set_title("Permutation importance on val (BA drop)")
    for ax in axes:
        ax.grid(axis="x", alpha=0.3)
    save_fig(fig, path)


def plot_confusion(y, pred, path, title):
    """행 정규화 혼동행렬 (대각선 = 클래스별 재현율)."""
    cm = confusion_matrix(y, pred, labels=range(18), normalize="true")
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(18), ev.CLASSES, rotation=90, fontsize=7)
    ax.set_yticks(range(18), ev.CLASSES, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(im, fraction=0.046)
    save_fig(fig, path)


def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 실행 모드

def compare_val(args, frames, X, y, out, commit):
    """val로 5개 설정 × seed 3개를 비교한다. test는 열지 않는다."""
    n_val = len(y["val"])

    # 최소 기준선: train 클래스 비율로만 찍기
    dummy = DummyClassifier(strategy="prior").fit(X["train"], y["train"])
    s = ev.score(y["val"], predict_proba18(dummy, X["val"]))
    dummy_ba = balanced_accuracy_score(y["train"], dummy.predict(X["train"]))
    rows = [result_row("dummy_prior", "baseline", {"strategy": "prior"}, 42, "val", s,
                       0.0, 0.0, dummy_ba, commit, n_val)]
    rows[0].update(run_id="A_dummy_prior", model="DummyClassifier")

    probs, models = {}, {}
    for name, (cond, params) in CONFIGS.items():
        for seed in SEEDS:
            clf, fit_s, train_ba = fit_rf(params, seed, X["train"], y["train"], args.n_jobs)
            P, pred_s = timed_predict(clf, X["val"])
            s = ev.score(y["val"], P)
            rows.append(result_row(name, cond, params, seed, "val", s, fit_s, pred_s, train_ba, commit, n_val))
            probs[name, seed], models[name, seed] = P, clf
            ev.save_pred(frames["val"].id, P, ev.CLASSES, args.pred_dir / f"val_A_rf_{name}_seed{seed}.csv")
            print(f"{name:5s} seed{seed}  train_BA={train_ba:.4f}  val_BA={s['model_BA']:.4f}  "
                  f"val_F1={s['model_F1']:.4f}  fit={fit_s:.2f}s")

    res = pd.DataFrame(rows)
    res.to_csv(out / "val_results.csv", index=False)
    summary = summarize(res)
    summary.to_csv(out / "val_summary.csv")
    selected, best = select_config(summary)
    plot_comparison(summary, out / "fig_config_comparison.png")

    # 선택된 설정 분석: 특징 중요도, 혼동행렬, 오분류 목록
    imp = feature_importance([models[selected, s] for s in SEEDS], X["val"], y["val"], X["train"], args.n_jobs)
    imp.sort_values("mdi_mean", ascending=False).to_csv(out / "feature_importance.csv", index=False)
    plot_importance(imp, out / "fig_feature_importance.png")

    P = probs[selected, 42]
    plot_confusion(y["val"], P.argmax(1), out / "fig_confusion_val.png",
                   f"RF {selected} seed42, val (row-normalized)")
    error_table(frames["val"], X["val"], y["val"], P).to_csv(out / "val_errors_seed42.csv", index=False)

    meta = {
        "selected_config": selected, "best_mean_val_BA_config": best,
        "selection_rule": "max mean val model_BA; within best's seed std -> lower val log loss",
        "seeds": SEEDS, "configs": {k: v[1] for k, v in CONFIGS.items()},
        "constant_features_in_train": imp.feature[imp.constant_in_train].tolist(),
        "environment": env_info(args.n_jobs), "code_commit": commit, "test_opened": False,
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    pd.set_option("display.width", 200)
    print(summary.round(4).to_string())
    print(f"\n규칙으로 고른 설정: {selected}   val BA 평균 1위: {best}")
    print(f"저장 위치: {out}")


def final_test(args, frames, X, y, out, commit):
    """고정한 설정 하나로 test를 한 번만 평가한다. train만으로 학습 (C 기준선과 동일)."""
    name = args.final
    if name not in CONFIGS:
        raise SystemExit(f"--final 값은 {list(CONFIGS)} 중 하나여야 합니다")
    target = out / f"test_results_{name}.csv"
    if target.exists() and not args.force:
        raise SystemExit(f"{target} 파일이 이미 있습니다: test는 이미 한 번 평가했습니다. test로 재튜닝하지 마세요.")

    cond, params = CONFIGS[name]
    rows = []
    for seed in SEEDS:
        clf, fit_s, train_ba = fit_rf(params, seed, X["train"], y["train"], args.n_jobs)
        for split in ["val", "test"]:
            P, pred_s = timed_predict(clf, X[split])
            s = ev.score(y[split], P)
            rows.append(result_row(name, cond, params, seed, split, s, fit_s, pred_s, train_ba,
                                   commit, len(y[split])))
            ev.save_pred(frames[split].id, P, ev.CLASSES, args.pred_dir / f"{split}_A_rf_{name}_seed{seed}.csv")
            if seed == 42 and split == "test":
                plot_confusion(y["test"], P.argmax(1), out / f"fig_confusion_test_{name}.png",
                               f"RF {name} seed42, test (row-normalized)")

    res = pd.DataFrame(rows)
    res.to_csv(target, index=False)
    print(res.groupby("split")[["model_BA", "accuracy", "macro_f1", "company_BA"]].agg(["mean", "std"]).round(4))

    # 질문 단위 bootstrap 95% 신뢰구간 (C의 공통 코드)
    data = ev.load_data(*(args.data_dir / f"{s}.csv" for s in ["train", "val", "test"]))
    run = ev.load_run([str(args.pred_dir / f"test_A_rf_{name}_seed{s}.csv") for s in SEEDS],
                      data, split="test", open_test=True)
    ci = ev.bootstrap_ci(run)
    ci.to_csv(out / f"test_ci_{name}.csv")
    print(ci)


# ---------------------------------------------------------------- 기타

def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def env_info(n_jobs):
    return {"python": platform.python_version(), "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "n_jobs": n_jobs, "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
            "scipy": scipy.__version__, "matplotlib": matplotlib.__version__}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/split", help="A의 분할 CSV 폴더")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results/course/A/random_forest", help="결과 저장 폴더")
    p.add_argument("--pred-dir", type=Path, default=ROOT / "preds", help="예측 확률 CSV 저장 폴더")
    p.add_argument("--n-jobs", type=int, default=-1, help="CPU 코어 수 (-1 = 전부)")
    p.add_argument("--final", help="test를 한 번 평가할 설정 이름 (예: d10)")
    p.add_argument("--force", action="store_true", help="기존 test 결과 덮어쓰기 (사용 자제)")
    p.add_argument("--commit", help=".git이 없을 때 기록할 코드 버전")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.pred_dir.mkdir(parents=True, exist_ok=True)

    frames = load_splits(args.data_dir, include_test=bool(args.final))  # test는 --final일 때만
    X = make_features(frames)
    y = {k: make_labels(df) for k, df in frames.items()}
    commit = args.commit or git_commit()

    if args.final:
        final_test(args, frames, X, y, args.out_dir, commit)
    else:
        compare_val(args, frames, X, y, args.out_dir, commit)


if __name__ == "__main__":
    main()
