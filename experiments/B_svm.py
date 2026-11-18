"""B: Linear SVM experiments (B-1 ~ B-4). Run from repo root: python -m experiments.B_svm

Train only on train, compare on val. Test is never opened here.
"""
import json
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from evaluate import CLASSES, M2C, confusion, save_pred, score
from features.build import load_splits, make_extractor
from features.stylometry import StylometricTransformer

SEED = 42
OUT = Path("results/course/B/svm")
PREDS = Path("preds")
CS = [0.1, 1, 10]
MAX_ITER = 5000  # non-convergence is recorded in the `converged` column


def make_feat(kind):
    raw = Pipeline([("raw", StylometricTransformer())])
    if kind == "style_raw":
        return raw
    # Centered scaling (common extractor uses with_mean=False): uncentered dense style columns next to
    # L2-normalized TF-IDF made liblinear take ~173 s per combined fit on Colab.
    style = Pipeline([("raw", StylometricTransformer()), ("scale", StandardScaler())])
    if kind == "style":
        return style
    if kind == "tfidf":
        return make_extractor("tfidf")
    return FeatureUnion([("style", style), ("tfidf", make_extractor("tfidf"))])


def make_pipe(kind, C):
    feat = make_feat(kind)
    return Pipeline([("feat", feat), ("svm", LinearSVC(C=C, random_state=SEED, max_iter=MAX_ITER))])


def raw_metrics(y, pred):
    return {"BA": balanced_accuracy_score(y, pred), "acc": accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro", labels=range(18), zero_division=0)}


def run(kind, C, data):
    print(f"start {kind} C={C} ({time.strftime('%H:%M:%S')})", flush=True)
    (Xtr, ytr, gtr), (Xva, yva) = data["train"], data["val"]
    name = f"B_svm_{kind}_C{C}"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipe = make_pipe(kind, C)
        t0 = time.perf_counter(); pipe.fit(Xtr, ytr); fit_s = time.perf_counter() - t0
        print(f"  fit {fit_s:.1f}s, now 5-fold calibration", flush=True)
        t0 = time.perf_counter(); dva = pipe.decision_function(Xva); pred_s = time.perf_counter() - t0
        tr_raw = raw_metrics(ytr, pipe.decision_function(Xtr).argmax(1))
        va_raw = raw_metrics(yva, dva.argmax(1))
        # Probabilities: sigmoid calibration, folds grouped by prompt_id; features refit inside each fold.
        folds = list(GroupKFold(n_splits=5).split(Xtr, ytr, gtr))
        cal = CalibratedClassifierCV(make_pipe(kind, C), method="sigmoid", cv=folds).fit(Xtr, ytr)
        P = cal.predict_proba(Xva)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    assert list(pipe.classes_) == list(range(18)) == list(cal.classes_)
    save_pred(data["val_ids"], P, CLASSES, PREDS / f"val_{name}_seed{SEED}.csv")
    s = score(yva, P)
    row = {"run_id": name, "owner": "B", "model": "LinearSVC", "feature_set": kind,
           "preprocessing": PREPROC[kind], "params": json.dumps({"C": C, "max_iter": MAX_ITER}),
           "seed": SEED, "split": "val", "n_samples": len(yva),
           "model_BA": s["model_BA"], "accuracy": s["model_Acc"], "macro_f1": s["model_F1"],
           "company_BA": s["company_BA"], "company_method": "prob_sum(calibrated)",
           "fit_seconds": round(fit_s, 2), "predict_seconds": round(pred_s, 3), "code_commit": COMMIT,
           "svm_train_BA": tr_raw["BA"], "svm_val_BA": va_raw["BA"], "svm_val_acc": va_raw["acc"],
           "svm_val_macro_f1": va_raw["macro_f1"], "converged": converged,
           **{f"cal_{k}": v for k, v in s.items()}}
    return row, pipe, dva


PREPROC = {"style": "32 stylometry -> StandardScaler (centered), fit on train",
           "style_raw": "32 stylometry, NO scaling",
           "tfidf": "char 3-5gram TF-IDF (max 30000, min_df=2, sublinear), fit on train",
           "combined": "centered-scaled stylometry + char TF-IDF (FeatureUnion), fit on train"}


def save_confusion(run_meta, name):
    cm = confusion(run_meta, "model")
    cm.to_csv(OUT / f"confusion_{name}.csv")
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(cm.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(18), CLASSES, rotation=90, fontsize=7)
    ax.set_yticks(range(18), CLASSES, fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{name} (val, row-normalized)")
    fig.tight_layout(); fig.savefig(OUT / f"confusion_{name}.png", dpi=120); plt.close(fig)


FINAL = [("style", 0.1), ("combined", 0.1)]  # fixed from val before test (docs/experiments/B-svm.md 8절)


def save_test_preds():
    """Refit the fixed final configs on train (same calibration folds) and save test probabilities. No scoring."""
    frames = load_splits("data/split", include_test=True)
    tr, te = frames["train"], frames["test"].sort_values("id").reset_index(drop=True)
    X, y = tr.text.to_numpy(), tr.model.map({m: i for i, m in enumerate(CLASSES)}).to_numpy()
    folds = list(GroupKFold(n_splits=5).split(X, y, tr.prompt_id.to_numpy()))
    for kind, C in FINAL:
        path = PREDS / f"test_B_svm_{kind}_C{C}_seed{SEED}.csv"
        if path.exists():
            print(f"skip {path}", flush=True); continue
        print(f"test preds {kind} C={C} ({time.strftime('%H:%M:%S')})", flush=True)
        cal = CalibratedClassifierCV(make_pipe(kind, C), method="sigmoid", cv=folds).fit(X, y)
        save_pred(te.id, cal.predict_proba(te.text.to_numpy()), CLASSES, path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PREDS.mkdir(exist_ok=True)
    if "--test" in sys.argv:
        return save_test_preds()
    frames = load_splits("data/split")
    tr, va = frames["train"], frames["val"]
    idx = {m: i for i, m in enumerate(CLASSES)}
    va = va.sort_values("id").reset_index(drop=True)
    data = {"train": (tr.text.to_numpy(), tr.model.map(idx).to_numpy(), tr.prompt_id.to_numpy()),
            "val": (va.text.to_numpy(), va.model.map(idx).to_numpy()), "val_ids": va.id}
    yva = data["val"][1]

    # B-1: input check (constant features on train)
    style = StylometricTransformer().fit(tr.text)
    S = pd.DataFrame(style.transform(tr.text), columns=style.get_feature_names_out())
    constant = S.columns[S.nunique() == 1].tolist()
    print(f"[B-1] style features={S.shape[1]}, constant on train={len(constant)}: {constant}", flush=True)

    # Resumable: every finished config is saved at once, and a rerun skips run_ids already in metrics.csv.
    metrics = OUT / "metrics.csv"
    rows, fitted = (pd.read_csv(metrics).to_dict("records") if metrics.exists() else []), {}
    done = lambda rid: any(r["run_id"] == rid for r in rows)
    save = lambda: pd.DataFrame(rows).to_csv(metrics, index=False)
    if not done("B_dummy_prior"):  # minimum baseline: class prior
        dummy = DummyClassifier(strategy="prior").fit(data["train"][0], data["train"][1])
        sd = score(yva, dummy.predict_proba(data["val"][0]))
        rows.append({"run_id": "B_dummy_prior", "owner": "B", "model": "DummyClassifier(prior)", "feature_set": "none",
                     "preprocessing": "none", "params": "{}", "seed": SEED, "split": "val", "n_samples": len(yva),
                     "model_BA": sd["model_BA"], "accuracy": sd["model_Acc"], "macro_f1": sd["model_F1"],
                     "company_BA": sd["company_BA"], "company_method": "prob_sum", "code_commit": COMMIT,
                     **{f"cal_{k}": v for k, v in sd.items()}})
        save()

    # B-2: common style input, C grid; B-3: same grid on tfidf / combined
    for kind in ["style", "tfidf", "combined"]:
        for C in CS:
            if done(f"B_svm_{kind}_C{C}"):
                print(f"skip B_svm_{kind}_C{C} (already in metrics.csv)", flush=True)
                continue
            row, pipe, dva = run(kind, C, data)
            rows.append(row); fitted[(kind, C)] = (pipe, dva); save()
            print(f"{row['run_id']:28s} trainBA={row['svm_train_BA']:.3f} valBA={row['svm_val_BA']:.3f} "
                  f"calBA={row['model_BA']:.3f} fit={row['fit_seconds']}s conv={row['converged']}", flush=True)
    c_star = best_style_c(pd.DataFrame(rows))
    print(f"[B-2] C* (best style val BA) = {c_star}", flush=True)

    # B-4: scaling evidence at C*
    if not done(f"B_svm_style_raw_C{c_star}"):
        row, _, _ = run("style_raw", c_star, data)
        rows.append(row); save()
        print(f"{row['run_id']:28s} trainBA={row['svm_train_BA']:.3f} valBA={row['svm_val_BA']:.3f} conv={row['converged']}")
    analyze(pd.DataFrame(rows), fitted, data, va, constant)


def C_of(params):
    return json.loads(params).get("C")  # dummy baseline row has no C


def best_style_c(df):
    # ties (all style C gave equal val BA) -> stable sort keeps grid order, i.e. the smallest C / strongest regularization
    style = df[(df.model == "LinearSVC") & (df.feature_set == "style")]
    return float(C_of(style.sort_values("svm_val_BA", ascending=False, kind="stable").iloc[0].params))


def analyze(df, fitted, data, va, constant):
    yva = data["val"][1]
    c_star = best_style_c(df)
    for kind in ["style", "tfidf", "combined"]:
        if (kind, c_star) not in fitted:  # --resume: refit the uncalibrated SVM only (deterministic, seed fixed)
            print(f"refit {kind} C={c_star} for analysis", flush=True)
            pipe = make_pipe(kind, c_star).fit(*data["train"][:2])
            fitted[(kind, c_star)] = (pipe, pipe.decision_function(data["val"][0]))

    # B-3 table: fixed C* across inputs vs. per-input retuned C
    fixed = df[(df.model == "LinearSVC") & (df.params.map(C_of) == c_star)]
    best = df[df.model == "LinearSVC"].sort_values("svm_val_BA", ascending=False).groupby("feature_set").head(1)
    cols = ["run_id", "feature_set", "params", "svm_train_BA", "svm_val_BA", "svm_val_acc", "svm_val_macro_f1",
            "model_BA", "accuracy", "macro_f1", "company_BA", "cal_model_LogLoss", "cal_sibling_Acc"]
    fixed[cols].to_csv(OUT / "feature_compare_fixedC.csv", index=False)
    best[cols].to_csv(OUT / "feature_compare_retunedC.csv", index=False)

    # B-4: confusion + misclassified val rows for each input at C*
    for kind in ["style", "tfidf", "combined"]:
        pipe, dva = fitted[(kind, c_star)]
        name = f"{kind}_C{c_star:g}"
        P1 = np.eye(18)[dva.argmax(1)]  # hard SVM decision, for confusion only
        save_confusion({"y": yva, "P": P1[None]}, name)
        pred = dva.argmax(1)
        err = va.assign(true=va.model, pred=[CLASSES[i] for i in pred],
                        same_company=M2C[pred] == M2C[yva], text_head=va.text.str[:300])
        err[pred != yva][["id", "prompt_id", "true", "pred", "same_company", "text_head"]] \
            .to_csv(OUT / f"errors_{name}.csv", index=False)

    # Interpretation: strongest stylometry coefficients and char n-grams per class
    for kind in ["style", "tfidf"]:
        pipe, _ = fitted[(kind, c_star)]
        names = np.array(pipe.named_steps["feat"].get_feature_names_out())
        coef = pipe.named_steps["svm"].coef_
        top = {CLASSES[k]: [f"{names[j]} ({coef[k, j]:+.2f})" for j in np.argsort(-coef[k])[:8]] for k in range(18)}
        pd.DataFrame(top).T.to_csv(OUT / f"top_coef_{kind}_C{c_star:g}.csv")

    meta = {"seed": SEED, "C_grid": CS, "C_star": c_star, "constant_style_features_train": constant,
            "calibration": "CalibratedClassifierCV(sigmoid, GroupKFold(5) by prompt_id on train, ensemble=True)",
            "selection": "C* = best LinearSVC val model BA (argmax of decision_function) on style input",
            "commit": COMMIT, "versions": {"python": platform.python_version(), "numpy": np.__version__,
                                           "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
            "platform": platform.platform(), "processor": platform.processor(), "test_opened": False}
    (OUT / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


COMMIT = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() + \
    ("-dirty" if subprocess.run(["git", "status", "--porcelain", "--", "experiments", "features", "evaluate.py"],
                                capture_output=True, text=True).stdout.strip() else "")

if __name__ == "__main__":
    main()
