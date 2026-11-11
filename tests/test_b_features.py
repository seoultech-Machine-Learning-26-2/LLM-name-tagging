"""Check feature semantics, leakage prevention and split alignment."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy import sparse

from features.build import build, load_splits, make_extractor
from features.stylometry import FEATURE_NAMES, StylometricTransformer, text_features


class FeatureTests(unittest.TestCase):
    def test_counts_and_empty(self):
        row = dict(zip(FEATURE_NAMES, text_features("Hello, world!")))
        self.assertEqual(row["char_count"], 13)
        self.assertEqual(row["word_count"], 2)
        self.assertAlmostEqual(row["comma_per_100_chars"], 100 / 13)
        self.assertEqual(row["sentence_count"], 1)
        self.assertEqual(len(row), 32)
        self.assertTrue(np.isfinite(text_features("")).all())
        self.assertEqual(sum(text_features("")), 0)
        self.assertEqual(dict(zip(FEATURE_NAMES, text_features("안녕 세계。")))["word_count"], 2)
        with self.assertRaises(TypeError):
            text_features(None)

    def test_markdown_and_order(self):
        a, b = "# Title\n- **bold**\n1. item\n```py\n```", "short"
        x = StylometricTransformer().fit_transform([a, b])
        self.assertEqual(x[1, 0], len(b))
        row = dict(zip(FEATURE_NAMES, x[0]))
        self.assertEqual(row["heading_per_100_lines"], 20)
        self.assertEqual(row["bullet_per_100_lines"], 20)
        self.assertEqual(row["code_fence_per_100_lines"], 40)

    def test_train_only_fit(self):
        train = ["aaaa common", "aaaa common", "bbbb common"]
        for kind in ["style", "tfidf", "combined"]:
            extractor = make_extractor(kind).fit(train)
            names = extractor.get_feature_names_out().copy()
            before = extractor.transform(train)
            extractor.transform(["VALIDATION_SECRET_zzzz"])
            after = extractor.transform(train)
            np.testing.assert_array_equal(names, extractor.get_feature_names_out())
            np.testing.assert_allclose(before.toarray() if sparse.issparse(before) else before,
                                       after.toarray() if sparse.issparse(after) else after)
            self.assertFalse(any("zzz" in x for x in names))

    def test_export_alignment_and_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for split, ids, prompts in [("train", [9, 2], [1, 2]), ("val", [8, 3], [3, 4])]:
                pd.DataFrame({"id": ids, "prompt_id": prompts,
                              "text": ["First sentence.", "Second text!"],
                              "model": ["a", "b"]}).to_csv(root / f"{split}.csv", index=False)
            # No test.csv exists: default build must succeed without opening test.
            build(root, root / "out", "style")
            self.assertEqual(pd.read_csv(root / "out/style/val_rows.csv").id.tolist(), [8, 3])
            self.assertEqual(sparse.load_npz(root / "out/style/train.npz").shape, (2, 32))
            saved = joblib.load(root / "out/style/extractor.joblib")
            np.testing.assert_allclose(saved.transform(pd.read_csv(root / "train.csv").text),
                                       sparse.load_npz(root / "out/style/train.npz").toarray())
            bad = pd.read_csv(root / "val.csv")
            bad.loc[0, "prompt_id"] = 1
            bad.to_csv(root / "val.csv", index=False)
            with self.assertRaises(ValueError):
                load_splits(root)


if __name__ == "__main__":
    unittest.main()
