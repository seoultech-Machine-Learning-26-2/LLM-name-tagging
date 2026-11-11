"""python -m features.build --kind style|tfidf|combined [--include-test]."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from .stylometry import StylometricTransformer


def make_extractor(kind="style"):
    style = Pipeline([("raw", StylometricTransformer()),
                      ("scale", StandardScaler(with_mean=False))])
    # Character n-grams preserve case, punctuation and internal whitespace.
    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=False,
                           min_df=2, max_features=30000, sublinear_tf=True,
                           strip_accents=None, dtype=np.float64)
    if kind == "style":
        return style
    if kind == "tfidf":
        return tfidf
    if kind == "combined":
        return FeatureUnion([("style", style), ("tfidf", tfidf)])
    raise ValueError(f"Unknown feature kind: {kind}")


def load_splits(data_dir, include_test=False):
    """Validate split metadata; test is only opened on explicit request."""
    frames = {}
    for split in ["train", "val"] + (["test"] if include_test else []):
        df = pd.read_csv(Path(data_dir) / f"{split}.csv")
        required = ["id", "prompt_id", "text", "model"]
        if not set(required).issubset(df.columns) or df[required].isna().any().any():
            raise ValueError(f"{split}: missing required columns/values")
        if not df.id.is_unique or not df.text.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError(f"{split}: duplicate id or empty/non-string text")
        if split != "train" and set(df.model) != set(frames["train"].model):
            raise ValueError(f"{split}: class set differs from train")
        for name, other in frames.items():
            if set(df.id) & set(other.id) or set(df.prompt_id) & set(other.prompt_id):
                raise ValueError(f"{name}/{split}: id or prompt overlap")
        frames[split] = df
    return frames


def build(data_dir, output_dir, kind, include_test=False):
    frames = load_splits(data_dir, include_test)
    output = Path(output_dir) / kind
    output.mkdir(parents=True, exist_ok=True)
    extractor = make_extractor(kind)
    extractor.fit(frames["train"].text)  # the ONLY fit; no validation/test fitting
    names = extractor.get_feature_names_out().tolist()
    manifest = {"kind": kind, "fit_split": "train", "feature_count": len(names),
                "classes": sorted(frames["train"].model.unique()), "splits": {},
                "versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                             "scipy": scipy.__version__}}
    for split, df in frames.items():
        matrix = sparse.csr_matrix(extractor.transform(df.text))
        if not np.isfinite(matrix.data).all():
            raise ValueError("Nonfinite feature value")
        sparse.save_npz(output / f"{split}.npz", matrix)
        df[["id", "prompt_id", "model"]].to_csv(output / f"{split}_rows.csv", index=False)
        manifest["splits"][split] = {
            "shape": list(matrix.shape),
            "sha256": hashlib.sha256((Path(data_dir) / f"{split}.csv").read_bytes()).hexdigest()}
    joblib.dump(extractor, output / "extractor.joblib")
    (output / "feature_names.json").write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/split")
    parser.add_argument("--output-dir", default="artifacts/features")
    parser.add_argument("--kind", choices=["style", "tfidf", "combined"], default="style")
    parser.add_argument("--include-test", action="store_true")
    args = parser.parse_args()
    build(args.data_dir, args.output_dir, args.kind, args.include_test)
