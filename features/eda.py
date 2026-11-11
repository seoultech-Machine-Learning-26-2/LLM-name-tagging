"""Train-only descriptive analysis: python -m features.eda."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .stylometry import FEATURE_NAMES, StylometricTransformer


def run(data_dir, output_dir):
    source = Path(data_dir) / "train.csv"
    df = pd.read_csv(source)
    if df[["id", "prompt_id", "text", "model"]].isna().any().any():
        raise ValueError("Missing train values")
    if not df.id.is_unique or df.text.str.strip().eq("").any():
        raise ValueError("Duplicate id or empty train text")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    values = pd.DataFrame(StylometricTransformer().fit_transform(df.text), columns=FEATURE_NAMES)
    full = pd.concat([df[["id", "prompt_id", "model"]].reset_index(drop=True), values], axis=1)
    full.to_csv(out / "train_style_features.csv", index=False)
    stats = full.groupby("model")[list(FEATURE_NAMES)].agg(["mean", "median", "std", "min", "max"])
    stats.columns = [f"{a}_{b}" for a, b in stats.columns]
    stats.insert(0, "count", full.groupby("model").size())
    stats.to_csv(out / "train_model_summary.csv")
    summary = ["| 모델 | 글 수 | 평균 문자 수 | 평균 단어 수 | 문자 100개당 부호 수 |",
               "| --- | ---: | ---: | ---: | ---: |"]
    for model, row in stats.iterrows():
        summary.append(f"| {model} | {int(row['count'])} | {row['char_count_mean']:.1f} | "
                       f"{row['word_count_mean']:.1f} | {row['punctuation_per_100_chars_mean']:.2f} |")
    (out / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    # Subtract each prompt's mean to examine differences within the same question.
    centered = values - full.groupby("prompt_id")[list(FEATURE_NAMES)].transform("mean")
    centered.assign(model=df.model).groupby("model").mean().to_csv(out / "train_prompt_centered.csv")
    metrics = ["char_count", "word_count", "punctuation_per_100_chars", "em_dash_per_100_chars"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), layout="constrained")
    models = sorted(df.model.unique())
    for ax, metric in zip(axes.flat, metrics):
        groups = [full.loc[full.model == m, metric].to_numpy() for m in models]
        ax.boxplot(groups, orientation="horizontal", tick_labels=models, showfliers=False)
        ax.set_title(metric + " (train; outliers hidden)")
        ax.grid(axis="x", alpha=0.2)
    fig.savefig(out / "train_distributions.png", dpi=130)
    plt.close(fig)
    selected = ["char_count", "word_count", "mean_word_length", "unique_word_ratio",
                "punctuation_per_100_chars", "comma_per_100_chars",
                "em_dash_per_100_chars", "semicolon_per_100_chars"]
    means = full.groupby("model")[selected].mean()
    z = (means - means.mean()) / means.std(ddof=0).replace(0, 1)
    fig, ax = plt.subplots(figsize=(12, 9), layout="constrained")
    im = ax.imshow(z.to_numpy(), cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_yticks(range(len(models)), models)
    ax.set_xticks(range(len(selected)), selected, rotation=40, ha="right")
    ax.set_title("Train model means: standardized across 18 models")
    fig.colorbar(im, ax=ax, label="z-score (clipped colors at +/-2.5)")
    fig.savefig(out / "train_style_heatmap.png", dpi=130)
    plt.close(fig)
    metadata = {"scope": "train only; validation/test not opened", "rows": len(df),
                "prompts": int(df.prompt_id.nunique()), "models": len(models),
                "duplicate_texts": int(df.text.duplicated().sum()),
                "feature_count": len(FEATURE_NAMES),
                "constant_features": {c: float(values[c].iloc[0]) for c in FEATURE_NAMES
                                      if values[c].nunique() == 1},
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "matplotlib": matplotlib.__version__}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(stats[["count", "char_count_mean", "word_count_mean", "punctuation_per_100_chars_mean"]].round(2).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data/split")
    p.add_argument("--output-dir", default="results/b_eda")
    args = p.parse_args()
    run(args.data_dir, args.output_dir)
