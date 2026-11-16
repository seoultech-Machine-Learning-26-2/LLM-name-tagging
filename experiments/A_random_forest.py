"""[A] 공통 문체 특징 32개로 학습하는 Random Forest 실험 (작업 A-2 ~ A-4).

사용법 (저장소 최상위 폴더에서 실행):
    python experiments/A_random_forest.py                  # val로 설정 비교 (test는 열지 않음)
    python experiments/A_random_forest.py --final n100     # 설정을 고정한 뒤 test를 한 번만 평가

따르는 규칙 (docs/next-tasks-A-B-C-D.md):
- A의 분할 데이터를 그대로 쓰고, B의 문체 특징 32개를 입력으로 쓴다.
  트리 모델은 값의 크기(스케일)에 영향을 받지 않으므로 스케일링 없이 원래 값을 쓴다.
- 학습은 train에서만 하고, 설정 비교는 val로 한다. test는 설정을 고정한 뒤 --final로만 연다.
- 한 번에 조건 하나만 바꾼다: 트리 수 50/100/300 (깊이 제한 없음), 깊이 5/10/None (트리 100개).
- seed는 42/43/44 세 개를 쓴다. 설정 선택 기준은 val model_BA 평균이 가장 높은 설정이며,
  1위와의 차이가 1위 설정의 seed 표준편차 이내인 설정들 중에서는 val log loss가 낮은 쪽을 고른다
  (C의 기존 규칙).
- 지표는 C가 만든 공통 평가 코드 evaluate.py로 계산한다.
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

# 저장소 최상위 폴더를 import 경로에 추가한다 (evaluate.py, features/ 를 불러오기 위해)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")  # 화면 없이 그림 파일만 저장하는 모드 (Kaggle·서버용)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

import evaluate as ev                                   # C의 공통 평가 코드
from features.build import load_splits                  # B의 분할 CSV 로드·검사 함수
from features.stylometry import FEATURE_NAMES, StylometricTransformer  # B의 문체 특징 32개

# 실험에 쓸 seed 3개
SEEDS = [42, 43, 44]

# 기본 설정 (계획 문서의 시작 설정, 최적값이라는 뜻은 아님)
BASE = dict(n_estimators=100, max_depth=None, min_samples_leaf=1)

# 비교할 설정 목록: 이름 -> (바꾼 조건, 하이퍼파라미터)
# 기본 설정(n100)에서 조건을 하나씩만 바꾼다. n100은 두 비교에서 공통으로 재사용한다.
CONFIGS = {
    "n50": ("n_estimators", {**BASE, "n_estimators": 50}),    # 트리 수 50
    "n100": ("base", dict(BASE)),                              # 기본 설정
    "n300": ("n_estimators", {**BASE, "n_estimators": 300}),  # 트리 수 300
    "d5": ("max_depth", {**BASE, "max_depth": 5}),            # 최대 깊이 5
    "d10": ("max_depth", {**BASE, "max_depth": 10}),          # 최대 깊이 10
}


def git_commit():
    """현재 코드의 git 커밋 해시를 가져온다. .git 폴더가 없으면 'unknown'을 반환한다."""
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def env_info(n_jobs):
    """실행 환경과 라이브러리 버전을 기록한다 (보고서 첫 문항에 필요)."""
    return {"python": platform.python_version(), "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "n_jobs": n_jobs, "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
            "scipy": scipy.__version__, "matplotlib": matplotlib.__version__}


def features(frames):
    """각 분할의 text에서 문체 특징 32개를 계산한다.

    특징은 글 하나만 보고 계산하는 고정된 공식이라 데이터에서 학습하는 값이 없다.
    따라서 val/test 정보가 학습에 새어 들어가지 않는다.
    """
    st = StylometricTransformer().fit(frames["train"].text)
    return {k: np.asarray(st.transform(df.text), dtype=float) for k, df in frames.items()}


def labels(df):
    """모델 이름을 숫자 라벨(0~17)로 바꾼다. 순서는 evaluate.CLASSES(알파벳순)와 같다."""
    return df.model.map({m: i for i, m in enumerate(ev.CLASSES)}).to_numpy()


def proba18(model, X):
    """예측 확률을 (글 수, 18) 행렬로 만든다. 열 순서는 evaluate.CLASSES와 같다.

    sklearn의 classes_ 에는 학습에 등장한 라벨 번호가 들어 있으므로 그 위치에 확률을 채운다.
    """
    P = np.zeros((len(X), 18))
    P[:, model.classes_] = model.predict_proba(X)
    return P


def run_one(name, params, seed, X, y, n_jobs):
    """설정 하나 × seed 하나로 RF를 train에 학습하고 val 확률을 예측한다.

    반환값: 학습된 모델, val 확률, 학습 시간(초), 예측 시간(초), train BA
    """
    clf = RandomForestClassifier(**params, random_state=seed, n_jobs=n_jobs)
    t0 = time.perf_counter()
    clf.fit(X["train"], y["train"])          # 학습은 train만 사용
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    P_val = proba18(clf, X["val"])
    pred_s = time.perf_counter() - t0
    # train BA: train을 얼마나 외웠는지 확인하는 값 (val BA와의 차이로 과적합을 본다)
    train_ba = balanced_accuracy_score(y["train"], clf.predict(X["train"]))
    return clf, P_val, fit_s, pred_s, train_ba


def row(name, cond, params, seed, split, n, s, fit_s, pred_s, train_ba, commit):
    """결과표 한 줄을 만든다. 열 이름은 계획 문서의 통합 결과표 필수 열을 따른다."""
    return {"run_id": f"A_rf_{name}_seed{seed}", "owner": "A", "model": "RandomForest",
            "config": name, "changed": cond, "feature_set": "style32",
            "preprocessing": "raw (no scaling)", "params": json.dumps(params), "seed": seed,
            "split": split, "n_samples": n, "train_model_BA": train_ba,
            "model_BA": s["model_BA"], "accuracy": s["model_Acc"], "macro_f1": s["model_F1"],
            "log_loss": s["model_LogLoss"], "sibling_acc": s["sibling_Acc"],
            # company_BA: 18개 모델 확률을 개발사별로 합산한 뒤 개발사를 고르는 방식
            "company_BA": s["company_BA"], "company_method": "prob_sum",
            # company_BA_argmax: 18개 중 1위 모델의 개발사를 쓰는 방식 (위와 직접 비교 금지)
            "company_BA_argmax": s["company_BA_argmax"], "fit_seconds": fit_s,
            "predict_seconds": pred_s, "code_commit": commit}


def select(summary):
    """C의 기존 규칙으로 설정을 고른다.

    1) val BA 평균이 가장 높은 설정(best)을 찾는다.
    2) best와의 차이가 best의 seed 표준편차 이내인 설정들을 후보로 둔다.
    3) 후보 중 val log loss 평균이 가장 낮은 설정을 고른다.
    반환값: (규칙으로 고른 설정, val BA 평균 1위 설정)
    """
    best = summary.sort_values("val_BA_mean", ascending=False).iloc[0]
    close = summary[summary.val_BA_mean >= best.val_BA_mean - best.val_BA_std]
    return close.sort_values("val_logloss_mean").index[0], best.name


def plot_comparison(summary, out):
    """설정별 train/val BA 비교 그림 (왼쪽: 트리 수, 오른쪽: 깊이). 오차 막대는 seed 표준편차."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, keys, xs, title in [
        (axes[0], ["n50", "n100", "n300"], ["50", "100", "300"], "n_estimators (max_depth=None)"),
        (axes[1], ["d5", "d10", "n100"], ["5", "10", "None"], "max_depth (n_estimators=100)")]:
        s = summary.loc[keys]
        ax.errorbar(xs, s.train_BA_mean, yerr=s.train_BA_std, marker="o", capsize=3, label="train BA")
        ax.errorbar(xs, s.val_BA_mean, yerr=s.val_BA_std, marker="s", capsize=3, label="val BA")
        # 18개 중 무작위로 고를 때의 기대값 (참고선)
        ax.axhline(1 / 18, color="gray", ls=":", lw=1, label="uniform 1/18")
        ax.set_title(title)
        ax.set_xlabel(title.split()[0])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("model BA (18 classes), mean ± std over seeds 42/43/44")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_importance(imp, out):
    """특징 중요도 상위 20개 그림.

    왼쪽 MDI: 트리가 train에서 그 특징으로 분기할 때 불순도를 얼마나 줄였는지의 합.
    오른쪽 순열 중요도: val에서 그 특징 값을 섞었을 때 BA가 얼마나 떨어지는지.
    """
    imp = imp.sort_values("mdi_mean").tail(20)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    axes[0].barh(imp.feature, imp.mdi_mean, xerr=imp.mdi_std)
    axes[0].set_title("MDI importance (train, mean over seeds)")
    axes[1].barh(imp.feature, imp.perm_mean, xerr=imp.perm_std, color="tab:orange")
    axes[1].set_title("Permutation importance on val (BA drop)")
    for ax in axes:
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_confusion(y, pred, out, title):
    """혼동행렬 그림. 각 행(실제 모델)을 합이 1이 되도록 정규화한다 = 대각선이 클래스별 재현율."""
    cm = confusion_matrix(y, pred, labels=range(18), normalize="true")
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(18), ev.CLASSES, rotation=90, fontsize=7)
    ax.set_yticks(range(18), ev.CLASSES, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def compare_val(args, frames, X, y, out, commit):
    """val로 설정을 비교한다 (test는 열지 않는다). 결과표·그림·오분류 목록을 저장한다."""
    rows, P_store, models = [], {}, {}
    n_val = len(y["val"])

    # 최소 기준선: 클래스 사전확률 기준 (항상 train에서 가장 많은 클래스를 예측,
    # 확률은 train의 클래스 비율). 이 값보다 높아야 의미 있는 학습이다.
    d = DummyClassifier(strategy="prior").fit(X["train"], y["train"])
    s = ev.score(y["val"], proba18(d, X["val"]))
    rows.append(row("dummy_prior", "baseline", {"strategy": "prior"}, 42, "val", n_val, s, 0.0, 0.0,
                    balanced_accuracy_score(y["train"], d.predict(X["train"])), commit))
    rows[-1].update(run_id="A_dummy_prior", model="DummyClassifier")

    # 설정 5개 × seed 3개 = 15번 학습
    for name, (cond, params) in CONFIGS.items():
        for seed in SEEDS:
            clf, P, fit_s, pred_s, tr = run_one(name, params, seed, X, y, args.n_jobs)
            s = ev.score(y["val"], P)
            rows.append(row(name, cond, params, seed, "val", n_val, s, fit_s, pred_s, tr, commit))
            P_store[(name, seed)], models[(name, seed)] = P, clf
            # C의 save_pred 형식(id + 18개 확률)으로 val 예측 저장
            ev.save_pred(frames["val"].id, P, ev.CLASSES, args.pred_dir / f"val_A_rf_{name}_seed{seed}.csv")
            print(f"{name:5s} seed{seed}  train_BA={tr:.4f}  val_BA={s['model_BA']:.4f}  "
                  f"val_F1={s['model_F1']:.4f}  fit={fit_s:.2f}s")

    # 전체 실행 결과 저장 (seed별 한 줄씩)
    res = pd.DataFrame(rows)
    res.to_csv(out / "val_results.csv", index=False)

    # 설정별로 seed 3개의 평균·표준편차를 요약
    rf = res[res.model == "RandomForest"]
    summary = rf.groupby("config").agg(
        changed=("changed", "first"), train_BA_mean=("train_model_BA", "mean"),
        train_BA_std=("train_model_BA", "std"), val_BA_mean=("model_BA", "mean"),
        val_BA_std=("model_BA", "std"), val_acc_mean=("accuracy", "mean"),
        val_f1_mean=("macro_f1", "mean"), val_logloss_mean=("log_loss", "mean"),
        val_company_BA_mean=("company_BA", "mean"), val_sibling_mean=("sibling_acc", "mean"),
        fit_seconds_mean=("fit_seconds", "mean")).loc[list(CONFIGS)]
    # train BA - val BA: 클수록 train을 외워서 새 질문에 일반화하지 못한다는 뜻 (과적합)
    summary["gap_train_minus_val"] = summary.train_BA_mean - summary.val_BA_mean
    selected, best = select(summary)
    summary.to_csv(out / "val_summary.csv")
    plot_comparison(summary, out / "fig_config_comparison.png")

    # 선택된 설정의 특징 중요도 (분석용이며, 설정 선택에는 쓰지 않는다)
    # MDI는 train에서 계산되고, 순열 중요도는 val에서 계산한다.
    mdi = np.stack([models[(selected, s)].feature_importances_ for s in SEEDS])
    perm = np.stack([permutation_importance(models[(selected, s)], X["val"], y["val"],
                                            scoring="balanced_accuracy", n_repeats=10,
                                            random_state=s, n_jobs=args.n_jobs).importances_mean
                     for s in SEEDS])
    # train에서 값이 하나뿐인 특징(상수 특징)을 표시한다. B 문서상 32개 중 9개가 상수다.
    const = X["train"].std(0) == 0
    imp = pd.DataFrame({"feature": FEATURE_NAMES, "constant_in_train": const,
                        "mdi_mean": mdi.mean(0), "mdi_std": mdi.std(0),
                        "perm_mean": perm.mean(0), "perm_std": perm.std(0)})
    imp.sort_values("mdi_mean", ascending=False).to_csv(out / "feature_importance.csv", index=False)
    plot_importance(imp, out / "fig_feature_importance.png")

    # 선택된 설정(seed 42)의 val 혼동행렬과 오분류 목록
    P = P_store[(selected, 42)]
    pred = P.argmax(1)
    plot_confusion(y["val"], pred, out / "fig_confusion_val.png",
                   f"RF {selected} seed42, val (row-normalized)")
    v = frames["val"].reset_index(drop=True)
    # 틀린 글만 모아 저장한다. 확신이 높았던 오류부터 정렬해 오류 사례 분석에 쓴다.
    err = pd.DataFrame({"id": v.id, "prompt_id": v.prompt_id, "true": v.model,
                        "pred": [ev.CLASSES[i] for i in pred],
                        "pred_prob": P.max(1),                          # 틀린 예측에 준 확률
                        "true_prob": P[np.arange(len(P)), y["val"]],     # 정답에 준 확률
                        "same_company": ev.M2C[pred] == ev.M2C[y["val"]],  # 개발사는 맞혔는지
                        "char_count": X["val"][:, 0], "punct_per_100": X["val"][:, 12],
                        "em_dash_per_100": X["val"][:, FEATURE_NAMES.index("em_dash_per_100_chars")],
                        "text": v.text})[pred != y["val"]]
    err.sort_values("pred_prob", ascending=False).to_csv(out / "val_errors_seed42.csv", index=False)

    # 실행 정보 기록 (선택 결과, 설정, 상수 특징, 환경 버전, 커밋)
    meta = {"selected_config": selected, "best_mean_val_BA_config": best,
            "selection_rule": "max mean val model_BA; within best's seed std -> lower val log loss",
            "seeds": SEEDS, "configs": {k: v[1] for k, v in CONFIGS.items()},
            "constant_features_in_train": [f for f, c in zip(FEATURE_NAMES, const) if c],
            "environment": env_info(args.n_jobs), "code_commit": commit,
            "test_opened": False}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(summary.round(4).to_string())
    print(f"\n규칙으로 고른 설정: {selected}   val BA 평균 1위: {best}")
    print(f"저장 위치: {out}")


def final_test(args, frames, X, y, out, commit):
    """고정한 설정 하나로 test를 한 번만 평가한다.

    - 이미 test 결과 파일이 있으면 멈춘다 (test를 보고 다시 튜닝하는 것을 막기 위해).
    - C의 기준선과 같이 train만으로 학습한다 (train+val 재학습 안 함).
    """
    name = args.final
    if name not in CONFIGS:
        raise SystemExit(f"--final 값은 {list(CONFIGS)} 중 하나여야 합니다")
    target = out / f"test_results_{name}.csv"
    if target.exists() and not args.force:
        raise SystemExit(f"{target} 파일이 이미 있습니다: test는 이미 한 번 평가했습니다. test로 재튜닝하지 마세요.")
    cond, params = CONFIGS[name]
    rows = []
    for seed in SEEDS:
        clf = RandomForestClassifier(**params, random_state=seed, n_jobs=args.n_jobs)
        t0 = time.perf_counter()
        clf.fit(X["train"], y["train"])  # train만으로 학습
        fit_s = time.perf_counter() - t0
        tr = balanced_accuracy_score(y["train"], clf.predict(X["train"]))
        for split in ["val", "test"]:
            t0 = time.perf_counter()
            P = proba18(clf, X[split])
            pred_s = time.perf_counter() - t0
            s = ev.score(y[split], P)
            rows.append(row(name, cond, params, seed, split, len(y[split]), s, fit_s, pred_s, tr, commit))
            ev.save_pred(frames[split].id, P, ev.CLASSES, args.pred_dir / f"{split}_A_rf_{name}_seed{seed}.csv")
        if seed == 42:
            # 반복문의 마지막 split이 test이므로 여기서 P는 test 확률이다
            plot_confusion(y["test"], P.argmax(1), out / f"fig_confusion_test_{name}.png",
                           f"RF {name} seed42, test (row-normalized)")
    res = pd.DataFrame(rows)
    res.to_csv(target, index=False)
    print(res.groupby("split")[["model_BA", "accuracy", "macro_f1", "company_BA"]].agg(["mean", "std"]).round(4))
    # C의 공통 코드로 질문 단위 bootstrap 95% 신뢰구간 계산
    # (같은 질문의 응답은 함께 뽑아서, 서로 독립이 아닌 응답을 독립처럼 세지 않는다)
    data = ev.load_data(*(args.data_dir / f"{s}.csv" for s in ["train", "val", "test"]))
    run = ev.load_run([str(args.pred_dir / f"test_A_rf_{name}_seed{s}.csv") for s in SEEDS],
                      data, split="test", open_test=True)
    ci = ev.bootstrap_ci(run)
    ci.to_csv(out / f"test_ci_{name}.csv")
    print(ci)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/split", help="A의 분할 CSV 폴더")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results/course/A/random_forest",
                   help="결과표·그림 저장 폴더")
    p.add_argument("--pred-dir", type=Path, default=ROOT / "preds", help="예측 확률 CSV 저장 폴더")
    p.add_argument("--n-jobs", type=int, default=-1, help="사용할 CPU 코어 수 (-1이면 전부)")
    p.add_argument("--final", help="test를 한 번 평가할 고정 설정 이름 (예: n100)")
    p.add_argument("--force", action="store_true", help="이미 있는 test 결과를 덮어쓴다 (사용 자제)")
    p.add_argument("--commit", help=".git이 없을 때 기록할 코드 버전 (예: Kaggle에 ZIP으로 올린 경우)")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.pred_dir.mkdir(parents=True, exist_ok=True)

    # test는 --final을 줬을 때만 불러온다
    frames = load_splits(args.data_dir, include_test=bool(args.final))
    X, y = features(frames), {k: labels(df) for k, df in frames.items()}
    commit = args.commit or git_commit()
    if args.final:
        final_test(args, frames, X, y, args.out_dir, commit)
    else:
        compare_val(args, frames, X, y, args.out_dir, commit)


if __name__ == "__main__":
    main()
