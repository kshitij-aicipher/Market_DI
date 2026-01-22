import argparse
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

from .utils import clean_prices, clean_percent, build_text_all, save_json, squeeze_df_for_tfidf

TARGET_CANDIDATES = [
    "discounted_price",
    "actual_price",
    "discount_percentage",
    "rating",
    "rating_count"
]

def parse_args():
    p = argparse.ArgumentParser(description="Train regression models for ALL found targets.")
    p.add_argument("--data_path", required=True, help="Path to input CSV")
    p.add_argument("--model_dir", default="models", help="Directory to save models")
    p.add_argument("--text_cols", default="about_product,review_title,review_content")
    p.add_argument("--cat_cols", default="category")
    return p.parse_args()

def _get_feature_conf(df, target, text_cols, cat_cols):
    all_cols = df.columns.tolist()
    num_candidates = [c for c in TARGET_CANDIDATES if c in all_cols]
    feature_num = [c for c in num_candidates if c != target]
    feature_cat = [c for c in cat_cols if c in all_cols and c != target]
    feature_text = ["text_all"]
    return feature_num, feature_cat, feature_text

def train_one_target(df, target, args, summary_dict):
    print(f"\n--- Training for Target: {target} ---")

    data = df.copy()
    data = data.dropna(subset=[target])

    if len(data) < 10:
        print(f"Skipping {target}: too few samples")
        return

    cat_cols_arg = [x.strip() for x in args.cat_cols.split(",") if x.strip()]
    feat_num, feat_cat, feat_text = _get_feature_conf(data, target, args.text_cols, cat_cols_arg)

    X = data[feat_num + feat_cat + feat_text]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    text_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="")),
        ("squeeze", FunctionTransformer(squeeze_df_for_tfidf)),
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1,1)))
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    transformers = []
    if feat_text: transformers.append(("text", text_pipe, feat_text))
    if feat_cat: transformers.append(("cat", cat_pipe, feat_cat))
    if feat_num: transformers.append(("num", num_pipe, feat_num))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=1.0))
    ])

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "rmse": float(mean_squared_error(y_test, preds, squared=False)),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds))
    }

    model_filename = f"{target}_model.joblib"
    save_path = os.path.join(args.model_dir, model_filename)
    joblib.dump(model, save_path)

    print(f"Saved {model_filename}")
    print(f"Metrics: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, R2={metrics['r2']:.3f}")

    summary_dict[target] = metrics

def main():
    args = parse_args()

    if args.data_path.endswith(".parquet"):
        df = pd.read_parquet(args.data_path)
    else:
        df = pd.read_csv(args.data_path)

    df = clean_prices(df, ["actual_price", "discounted_price"])
    df = clean_percent(df, "discount_percentage")

    for c in TARGET_CANDIDATES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    text_cols_list = [c.strip() for c in args.text_cols.split(",") if c.strip()]
    df = build_text_all(df, text_cols_list, "text_all")

    os.makedirs(args.model_dir, exist_ok=True)
    summary = {}

    for target in TARGET_CANDIDATES:
        if target in df.columns:
            if df[target].notna().sum() > 10:
                train_one_target(df, target, args, summary)
            else:
                print(f"Skipping {target}: mostly NaN")
        else:
            print(f"Target '{target}' not found")

    save_json(summary, os.path.join(args.model_dir, "training_summary.json"))
    print("\nDone. Summary saved to training_summary.json")

if __name__ == "__main__":
    main()
