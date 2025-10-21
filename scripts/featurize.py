import argparse
import warnings
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.pipeline import Pipeline
from sklearn import set_config
import joblib

warnings.filterwarnings("ignore")


def _safe_cols(df: pd.DataFrame, names):
    return [c for c in names if c in df.columns]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer using column names present in X."""

    default_num = _safe_cols(X, ["Age", "SibSp", "Parch", "Fare"])

    default_cat_nominal = _safe_cols(X, ["Pclass", "Sex", "Embarked"])

    optional_ordinal = _safe_cols(X, ["Title"])         
    optional_extra_cat = _safe_cols(X, ["Family_size"])   


    if "Family_size" in X.columns and pd.api.types.is_numeric_dtype(X["Family_size"]):
        if "Family_size" not in default_num:
            default_num.append("Family_size")
        if "Family_size" in optional_extra_cat:
            optional_extra_cat.remove("Family_size")


    numeric_features = default_num
    nominal_cats = default_cat_nominal + optional_extra_cat
    ordinal_features = optional_ordinal

    transformers = []
    if numeric_features:
        transformers.append(("scaling", MinMaxScaler(), numeric_features))

    if nominal_cats:
        transformers.append(("onehot", OneHotEncoder(handle_unknown="ignore"), nominal_cats))

    if ordinal_features:
        transformers.append(("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                             ordinal_features))


    kbins_targets = _safe_cols(X, ["Age", "Fare"])
    if kbins_targets:
        transformers.append(("kbins",
                             KBinsDiscretizer(n_bins=15, encode="ordinal", strategy="quantile"),
                             kbins_targets))


    preprocessor = ColumnTransformer(transformers, remainder="passthrough", verbose_feature_names_out=False)
    return preprocessor


def print_survival_correlations(df: pd.DataFrame):
    """Mimic your quick survival-rate by category printouts, but only for columns that exist."""
    cols = ["Embarked", "Pclass", "Sex", "Title", "Family_size"]
    for col in cols:
        if col in df.columns and "Survived" in df.columns:
            print("Survival Correlation by:", col)
            g = df.groupby(col)["Survived"].mean().reset_index().sort_values("Survived", ascending=False)
            print(g)
            print("-" * 10, "\n")


def create_pipeline(algo):
    """
    Keeps your original API:
    returns a Pipeline(preprocessor -> classifier)
    The preprocessor will be built later from the actual X_train columns.
    """

    dummy_pre = ColumnTransformer([("pass", "passthrough", [])], remainder="passthrough")
    return Pipeline([
        ("num_cat_transformation", dummy_pre),
        ("classifier", algo),
    ])


def main():
    parser = argparse.ArgumentParser(description="Featurize Titanic data after preprocessing")
    parser.add_argument("--train_path", type=str, required=True, help="Path to preprocessed training CSV")
    parser.add_argument("--test_path", type=str, required=True, help="Path to preprocessed test CSV")
    parser.add_argument("--out_train_X", type=str, default=None, help="Where to save transformed X_train (CSV)")
    parser.add_argument("--out_train_y", type=str, default=None, help="Where to save y_train (CSV)")
    parser.add_argument("--out_test_X", type=str, default=None, help="Where to save transformed X_test (CSV)")
    parser.add_argument("--preprocessor_path", type=str, default=None, help="Where to save fitted preprocessor (joblib)")
    args = parser.parse_args()


    for p in [args.out_train_X, args.out_train_y, args.out_test_X, args.preprocessor_path]:
        if p:
            Path(p).parent.mkdir(parents=True, exist_ok=True)

    print("Loading preprocessed data...")
    train = pd.read_csv(args.train_path)
    test = pd.read_csv(args.test_path)


    train = train.copy()
    test = test.copy()


    print_survival_correlations(train)


    if "Survived" not in train.columns:
        raise ValueError("Expected 'Survived' column in preprocessed train data.")

    y = train["Survived"].astype("int64")
    X = train.drop(columns=[c for c in ["Survived", "PassengerId"] if c in train.columns])
    X_test_raw = test.drop(columns=[c for c in ["Survived", "PassengerId"] if c in test.columns], errors="ignore")

    print("Building preprocessor...")
    preprocessor = build_preprocessor(X)

    print("Fitting preprocessor on train and transforming train/test...")
    X_trans = preprocessor.fit_transform(X)
    X_test_trans = preprocessor.transform(X_test_raw)


    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"f{i}" for i in range(X_trans.shape[1])]

    X_trans_df = pd.DataFrame(X_trans, columns=feature_names)
    X_test_trans_df = pd.DataFrame(X_test_trans, columns=feature_names)

    print("Shapes:")
    print("  X_train:", X_trans_df.shape, " y_train:", y.shape)
    print("  X_test :", X_test_trans_df.shape)


    if args.out_train_X:
        X_trans_df.to_csv(args.out_train_X, index=False)
        print(f"Saved transformed X_train to: {args.out_train_X}")
    if args.out_train_y:
        pd.DataFrame({"Survived": y}).to_csv(args.out_train_y, index=False)
        print(f"Saved y_train to: {args.out_train_y}")
    if args.out_test_X:
        X_test_trans_df.to_csv(args.out_test_X, index=False)
        print(f"Saved transformed X_test to: {args.out_test_X}")
    if args.preprocessor_path:
        joblib.dump(preprocessor, args.preprocessor_path)
        print(f"Saved fitted preprocessor to: {args.preprocessor_path}")

    print("Featurization complete.")


if __name__ == "__main__":
    main()
