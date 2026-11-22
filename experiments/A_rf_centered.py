"""[A] Random Forest + 질문 중심화 (2026-11-18 회의 Q-10 결정 2, 적용 규칙 Q-11).

사용법 (저장소 최상위 폴더에서):
    python experiments/A_rf_centered.py

- 모델 설정은 A_random_forest.py에서 val로 고른 d10(트리 100, 깊이 10)으로 고정하고 입력만 바꾼다.
    raw     : 문체 32개 그대로 (기존 d10과 같은 입력, 응답 단독 예측)
    sub     : X − 같은 prompt_id 응답들의 X 평균 (C XGBoost 방식)
    concat  : [X ; X − 질문 평균] (64개, C Qwen·B·D 방식)
- 질문 평균은 각 분할 안에서 같은 prompt_id 응답끼리만 계산한다. 정답(model)은 쓰지 않는다.
- seed 42/43/44마다 val·test 예측 파일을 따로 저장한다(Q-10 결정 4).
- test는 정답 열을 읽지 않는다(id·prompt_id·text만, C의 test_nolabel.csv와 같은 내용). 채점하지 않는다.
- 입력 선택은 val로만 한다(A_random_forest.py와 같은 규칙).
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

import evaluate as ev
from features.build import load_splits
from features.stylometry import FEATURE_NAMES, StylometricTransformer
from A_random_forest import (CONFIGS, SEEDS, env_info, error_table, feature_importance, fit_rf,
                             git_commit, make_labels, plot_confusion, plot_importance,
                             predict_proba18, save_fig)
import matplotlib.pyplot as plt

CONFIG = "d10"
VARIANTS = ["raw", "sub", "concat"]


def load_test_nolabel(data_dir):
    """test는 정답 열 없이 읽는다 (C test_nolabel.csv와 같은 내용)."""
    df = pd.read_csv(Path(data_dir) / "test.csv", usecols=["id", "prompt_id", "text"])
    if df[["id", "prompt_id", "text"]].isna().any().any() or not df.id.is_unique:
        raise ValueError("test: 결측 또는 중복 id")
    return df


def center(X, prompt_ids):
    """같은 prompt_id 응답들의 평균을 뺀다 (분할 안에서만, 정답 사용 안 함)."""
    means = pd.DataFrame(X).groupby(np.asarray(prompt_ids)).transform("mean").to_numpy()
    return X - means


def make_inputs(X, prompt_ids):
    C = center(X, prompt_ids)
    return {"raw": X, "sub": C, "concat": np.hstack([X, C])}


def feature_names(variant):
    if variant == "concat":
        return list(FEATURE_NAMES) + [f"{f}__centered" for f in FEATURE_NAMES]
    return list(FEATURE_NAMES)


def plot_variants(summary, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = list(summary.index)
    ax.errorbar(xs, summary.train_BA_mean, yerr=summary.train_BA_std, marker="o", capsize=3, label="train BA")
    ax.errorbar(xs, summary.val_BA_mean, yerr=summary.val_BA_std, marker="s", capsize=3, label="val BA")
    ax.axhline(1 / 18, color="gray", ls=":", lw=1, label="uniform 1/18")
    ax.set_title(f"RF {CONFIG}: input variants (mean ± std, seeds 42/43/44)")
    ax.set_xlabel("input")
    ax.set_ylabel("model BA (18 classes)")
    ax.grid(alpha=0.3)
    ax.legend()
    save_fig(fig, path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/split")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results/course/A/random_forest_centered")
    p.add_argument("--pred-dir", type=Path, default=ROOT / "preds")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--commit", help=".git이 없을 때 기록할 코드 버전")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.pred_dir.mkdir(parents=True, exist_ok=True)

    frames = load_splits(args.data_dir)                 # train·val (정답 포함)
    frames["test"] = load_test_nolabel(args.data_dir)   # test (정답 없음)
    st = StylometricTransformer().fit(frames["train"].text)
    X = {k: np.asarray(st.transform(df.text), dtype=float) for k, df in frames.items()}
    inputs = {k: make_inputs(X[k], frames[k].prompt_id) for k in frames}
    y = {k: make_labels(frames[k]) for k in ["train", "val"]}
    commit = args.commit or git_commit()
    params = CONFIGS[CONFIG][1]

    rows, probs, models = [], {}, {}
    for v in VARIANTS:
        for seed in SEEDS:
            clf, fit_s, train_ba = fit_rf(params, seed, inputs["train"][v], y["train"], args.n_jobs)
            t0 = time.perf_counter()
            P = predict_proba18(clf, inputs["val"][v])
            pred_s = time.perf_counter() - t0
            s = ev.score(y["val"], P)
            name = f"A_rf_{CONFIG}_{v}_seed{seed}"
            rows.append({
                "run_id": name, "owner": "A", "model": "RandomForest", "config": CONFIG, "input": v,
                "question_grouping": "응답 단독" if v == "raw" else "질문 묶음 이용",
                "feature_set": f"style32 ({v})", "n_features": inputs["train"][v].shape[1],
                "params": json.dumps(params), "seed": seed, "split": "val", "n_samples": len(P),
                "train_model_BA": train_ba, "model_BA": s["model_BA"], "accuracy": s["model_Acc"],
                "macro_f1": s["model_F1"], "log_loss": s["model_LogLoss"], "ece": s["model_ECE"],
                "sibling_acc": s["sibling_Acc"], "company_BA": s["company_BA"], "company_method": "prob_sum",
                "company_BA_argmax": s["company_BA_argmax"], "fit_seconds": fit_s, "predict_seconds": pred_s,
                "code_commit": commit})
            probs[v, seed], models[v, seed] = P, clf
            ev.save_pred(frames["val"].id, P, ev.CLASSES, args.pred_dir / f"val_{name}.csv")
            # test: 같은 질문 응답을 묶어 중심화한 입력으로 예측만 저장 (채점 없음)
            ev.save_pred(frames["test"].id, predict_proba18(clf, inputs["test"][v]), ev.CLASSES,
                         args.pred_dir / f"test_{name}.csv")
            print(f"{v:6s} seed{seed}  train_BA={train_ba:.4f}  val_BA={s['model_BA']:.4f}  "
                  f"val_F1={s['model_F1']:.4f}  fit={fit_s:.2f}s")

    res = pd.DataFrame(rows)
    res.to_csv(args.out_dir / "val_results.csv", index=False)
    summary = res.groupby("input").agg(
        n_features=("n_features", "first"),
        train_BA_mean=("train_model_BA", "mean"), train_BA_std=("train_model_BA", "std"),
        val_BA_mean=("model_BA", "mean"), val_BA_std=("model_BA", "std"),
        val_f1_mean=("macro_f1", "mean"), val_logloss_mean=("log_loss", "mean"), val_ece_mean=("ece", "mean"),
        val_company_BA_mean=("company_BA", "mean"), val_sibling_mean=("sibling_acc", "mean"),
    ).loc[VARIANTS]
    summary["gap_train_minus_val"] = summary.train_BA_mean - summary.val_BA_mean
    summary.to_csv(args.out_dir / "val_summary.csv")
    plot_variants(summary, args.out_dir / "fig_input_comparison.png")

    # 질문 묶음 이용 입력(sub, concat) 중에서 val 규칙으로 선택: BA 평균 1위와 std 이내면 log loss 낮은 쪽
    cand = summary.loc[["sub", "concat"]]
    best = cand.sort_values("val_BA_mean", ascending=False).iloc[0]
    close = cand[cand.val_BA_mean >= best.val_BA_mean - best.val_BA_std]
    selected = close.sort_values("val_logloss_mean").index[0]

    # 선택한 입력의 특징 중요도·혼동행렬·오분류 (분석용)
    imp = feature_importance([models[selected, s] for s in SEEDS], inputs["val"][selected], y["val"],
                             inputs["train"][selected], args.n_jobs)
    imp["feature"] = feature_names(selected)
    imp.sort_values("mdi_mean", ascending=False).to_csv(args.out_dir / "feature_importance.csv", index=False)
    plot_importance(imp, args.out_dir / "fig_feature_importance.png")
    P = probs[selected, 42]
    plot_confusion(y["val"], P.argmax(1), args.out_dir / "fig_confusion_val.png",
                   f"RF {CONFIG} {selected} seed42, val (row-normalized)")
    error_table(frames["val"], X["val"], y["val"], P).to_csv(args.out_dir / "val_errors_seed42.csv", index=False)

    # raw vs 선택 입력: 질문 단위 paired bootstrap (seed 3개 평균 확률, C 공통 코드)
    data = ev.load_data(*(args.data_dir / f"{s}.csv" for s in ["train", "val", "test"]))
    runs = {v: ev.load_run([str(args.pred_dir / f"val_A_rf_{CONFIG}_{v}_seed{s}.csv") for s in SEEDS], data, "val")
            for v in ["raw", selected]}
    for r in runs.values():
        r["P"] = r["P"].mean(0, keepdims=True)
    cmp = ev.compare(runs["raw"], runs[selected])
    cmp.to_csv(args.out_dir / f"val_compare_raw_vs_{selected}.csv")

    meta = {"config": CONFIG, "params": params, "variants": VARIANTS, "seeds": SEEDS,
            "selected_input": selected, "selection_rule": "sub·concat 중 val BA 평균 1위와 seed std 이내면 log loss 최소",
            "centering": "같은 분할 안 같은 prompt_id 응답들의 특징 평균을 뺌, 정답 사용 안 함",
            "test": "정답 열을 읽지 않고 예측 파일만 저장, 채점 없음",
            "environment": env_info(args.n_jobs), "code_commit": commit, "test_scored": False}
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    pd.set_option("display.width", 200)
    print(summary.round(4).to_string())
    print(f"\n질문 묶음 입력 중 선택: {selected}")
    print(f"raw → {selected} (seed 평균 확률, paired bootstrap):")
    print(cmp.round(4).to_string())
    print(f"저장 위치: {args.out_dir}, 예측: {args.pred_dir}/val_·test_A_rf_{CONFIG}_*")


if __name__ == "__main__":
    main()
