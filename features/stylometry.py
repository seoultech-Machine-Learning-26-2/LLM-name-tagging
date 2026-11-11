"""Deterministic, raw-text stylometry. No corpus statistics are fitted here."""
import re
import unicodedata

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

PUNCTUATION = {
    "period": ".", "comma": ",", "question": "?", "exclamation": "!",
    "colon": ":", "semicolon": ";", "hyphen": "-", "em_dash": "—",
    "parenthesis": "()", "quote": "\"'“”‘’", "asterisk": "*", "backtick": "`",
}
FEATURE_NAMES = (
    "char_count", "word_count", "sentence_count", "line_count",
    "mean_word_length", "mean_sentence_words", "word_length_std",
    "unique_word_ratio", "uppercase_ratio", "digit_ratio", "whitespace_ratio",
    "non_ascii_ratio", "punctuation_per_100_chars", "blank_line_ratio",
    "heading_per_100_lines", "bullet_per_100_lines", "numbered_per_100_lines",
    "code_fence_per_100_lines", "bold_per_100_chars", "link_per_100_chars",
) + tuple(f"{name}_per_100_chars" for name in PUNCTUATION)


def text_features(text):
    """32 features; word = Unicode letter/number sequence, apostrophes allowed.

    Sentence = nonempty fragment split by . ! ? 。！？ or newline.
    Ratios use raw character/line counts; empty text maps to finite zeros.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string; missing values must not become 'nan'")
    words = re.findall(r"[^\W_]+(?:['’][^\W_]+)*", text, flags=re.UNICODE)
    sentences = [s for s in re.split(r"[.!?。！？\n]+", text) if s.strip()]
    lines = text.splitlines() if text else []
    n, w, s, l = len(text), len(words), len(sentences), len(lines)
    nc, nl = max(n, 1), max(l, 1)
    lengths = [len(x) for x in words]
    values = [
        n, w, s, l, np.mean(lengths) if w else 0, w / max(s, 1),
        np.std(lengths) if w else 0, len({x.casefold() for x in words}) / max(w, 1),
        sum(c.isupper() for c in text) / nc, sum(c.isdigit() for c in text) / nc,
        sum(c.isspace() for c in text) / nc, sum(ord(c) > 127 for c in text) / nc,
        100 * sum(unicodedata.category(c).startswith("P") for c in text) / nc,
        sum(not line.strip() for line in lines) / nl,
        100 * len(re.findall(r"(?m)^\s{0,3}#{1,6}\s+", text)) / nl,
        100 * len(re.findall(r"(?m)^\s*[-+*]\s+", text)) / nl,
        100 * len(re.findall(r"(?m)^\s*\d+[.)]\s+", text)) / nl,
        100 * len(re.findall(r"(?m)^\s*(?:```|~~~)", text)) / nl,
        100 * len(re.findall(r"\*\*[^\n]+?\*\*", text)) / nc,
        100 * len(re.findall(r"\[[^\]\n]+\]\([^)\n]+\)", text)) / nc,
    ]
    values.extend(100 * sum(text.count(c) for c in chars) / nc
                  for chars in PUNCTUATION.values())
    return values


class StylometricTransformer(TransformerMixin, BaseEstimator):
    """scikit-learn pipeline compatible; preserves input row order."""

    def fit(self, X, y=None):
        self.n_features_out_ = len(FEATURE_NAMES)
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_out_")
        if isinstance(X, str):
            raise TypeError("Pass a sequence of texts, not one string")
        return np.asarray([text_features(text) for text in X], dtype=np.float64).reshape(
            -1, len(FEATURE_NAMES))

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURE_NAMES, dtype=object)
