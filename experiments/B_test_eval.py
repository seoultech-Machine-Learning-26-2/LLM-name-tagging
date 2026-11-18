"""B: one-time test scoring of the settings fixed on val (run once; refuses to overwrite).

    python -m experiments.B_test_eval
Candidates were fixed before this script existed: SVM style C=0.1, SVM combined C=0.1,
ModernBERT lam=0 / lam=0.1 (8 epochs, seeds 42-44, epoch chosen by val). No retuning after this.
"""
import json
from pathlib import Path

import pandas as pd

from evaluate import bootstrap_ci, compare, evaluate, load_data, load_run, template_table

OUT = Path("results/course/B/test")
RUNS = {"svm_style_C0.1": ["B_svm_style_C0.1_seed42"],
        "svm_combined_C0.1": ["B_svm_combined_C0.1_seed42"],
        "modernbert_lam0_ep8": [f"B_modernbert_lam0_ep8_seed{s}" for s in (42, 43, 44)],
        "modernbert_lam0.1_ep8": [f"B_modernbert_lam0.1_ep8_seed{s}" for s in (42, 43, 44)]}
PAIRS = [("svm_style_C0.1", "svm_combined_C0.1"), ("modernbert_lam0_ep8", "modernbert_lam0.1_ep8"),
         ("svm_combined_C0.1", "modernbert_lam0.1_ep8"), ("svm_combined_C0.1", "modernbert_lam0_ep8")]


def main():
    assert not OUT.exists(), f"{OUT} exists: test is scored once only"
    data = load_data("data/split/train.csv", "data/split/val.csv", "data/split/test.csv")
    val = {k: load_run([f"preds/val_{f}.csv" for f in v], data) for k, v in RUNS.items()}
    test = {k: load_run([f"preds/test_{f}.csv" for f in v], data, split="test", open_test=True) for k, v in RUNS.items()}
    OUT.mkdir(parents=True)
    rows = []
    for k in RUNS:
        for split, run in (("val", val[k]), ("test", test[k])):
            e, ci = evaluate(run), bootstrap_ci(run)
            rows.append({"run": k, "split": split, "n_seeds": len(run["P"]),
                         **{f"{m}_{s}": e.loc[m, s] for m in ("model_BA", "company_BA", "model_F1", "model_LogLoss",
                                                              "model_ECE", "sibling_Acc") for s in ("mean", "std")},
                         **{f"{m}_{c}": ci.loc[m, c] for m in ("model_BA", "company_BA") for c in ("ci_low", "ci_high")}})
        for lv in ("model", "company"):
            template_table(val[k], test[k], level=lv).to_csv(OUT / f"template_{lv}_{k}.csv", encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    cmp = pd.concat({f"{b} -> {n}": compare(test[b], test[n]) for b, n in PAIRS})
    cmp.to_csv(OUT / "compare_test.csv")
    (OUT / "meta.json").write_text(json.dumps({"runs": RUNS, "note": "scored once; settings fixed on val beforehand"}, indent=2))
    print(pd.DataFrame(rows)[["run", "split", "model_BA_mean", "model_BA_std", "company_BA_mean",
                              "model_BA_ci_low", "model_BA_ci_high"]].round(4).to_string(index=False))
    print(cmp.round(4).to_string())


if __name__ == "__main__":
    main()
