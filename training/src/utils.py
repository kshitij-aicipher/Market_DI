import re
import json
import numpy as np
import pandas as pd
from typing import List, Dict

_CURRENCY_RE = re.compile(r"[^0-9.]+")
_PERCENT_RE = re.compile(r"[^0-9.]+")

def clean_numeric_series(s: pd.Series) -> pd.Series:
    if s is None: return s
    s = s.astype(str).str.strip().replace({"": np.nan, "None": np.nan, "nan": np.nan})
    s = s.str.replace(_CURRENCY_RE, "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def clean_prices(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = clean_numeric_series(out[c])
    return out

def clean_percent(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    if col in out.columns:
        s = out[col].astype(str).str.strip().replace({"": np.nan})
        s = s.str.replace(_PERCENT_RE, "", regex=True)
        out[col] = pd.to_numeric(s, errors="coerce")
    return out

def build_text_all(df: pd.DataFrame, text_cols: List[str], out_col: str = "text_all") -> pd.DataFrame:
    out = df.copy()
    for c in text_cols:
        if c not in out.columns:
            out[c] = ""
    temp = out[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    out[out_col] = temp.str.replace(r"\s+", " ", regex=True).str.strip()
    return out

def save_json(obj: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def squeeze_df_for_tfidf(x):
    if hasattr(x, "to_numpy"): 
        x = x.to_numpy()
    return x.ravel()
