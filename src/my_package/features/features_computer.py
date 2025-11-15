import pandas as pd
from typing import Tuple
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, KBinsDiscretizer
from .base_features_computer import BaseFeaturesComputer


def _safe_cols(df: pd.DataFrame, names):
    return [c for c in names if c in df.columns]


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = _safe_cols(X, ["Age", "SibSp", "Parch", "Fare"])
    cat_nominal = _safe_cols(X, ["Pclass", "Sex", "Embarked"])
    ord_cols = _safe_cols(X, ["Title"])
    extra_cat = _safe_cols(X, ["Family_size"])

    if "Family_size" in X.columns and pd.api.types.is_numeric_dtype(X["Family_size"]):
        if "Family_size" not in num:
            num.append("Family_size")
        if "Family_size" in extra_cat:
            extra_cat.remove("Family_size")

    transformers = []
    if num:
        transformers.append(("num", MinMaxScaler(), num))
    if cat_nominal or extra_cat:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_nominal + extra_cat)
        )
    if ord_cols:
        transformers.append(
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ord_cols)
        )

    kbins_cols = _safe_cols(X, ["Age", "Fare"])
    if kbins_cols:
        transformers.append(
            ("kbins", KBinsDiscretizer(n_bins=15, encode="ordinal", strategy="quantile"), kbins_cols)
        )

    return ColumnTransformer(transformers, remainder="passthrough", verbose_feature_names_out=False)


def _print_survival_correlations(df: pd.DataFrame) -> None:
    for col in ["Embarked", "Pclass", "Sex", "Title", "Family_size"]:
        if col in df.columns and "Survived" in df.columns:
            print(f"Survival Correlation by: {col}")
            g = (
                df.groupby(col)["Survived"]
                .mean()
                .reset_index()
                .sort_values("Survived", ascending=False)
            )
            print(g)
            print("-" * 10, "\n")


class TitanicFeaturesComputer(BaseFeaturesComputer):
    """Compute train/test feature matrices for the Titanic dataset."""

    def __init__(self, test_df: pd.DataFrame):
        self.test_df = test_df
        self.preprocessor_: ColumnTransformer | None = None

    def compute(
        self, train_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, ColumnTransformer]:
        _print_survival_correlations(train_df)

        if "Survived" not in train_df.columns:
            raise ValueError("Expected 'Survived' column in preprocessed train data.")

        y = train_df["Survived"].astype("int64")
        X = train_df.drop(columns=[c for c in ["Survived", "PassengerId"] if c in train_df.columns])
        X_test_raw = self.test_df.drop(
            columns=[c for c in ["Survived", "PassengerId"] if c in self.test_df.columns],
            errors="ignore",
        )

        print("Building and fitting preprocessor...")
        pre = _build_preprocessor(X)
        X_tr = pre.fit_transform(X)
        X_te = pre.transform(X_test_raw)
        self.preprocessor_ = pre

        try:
            feature_names = pre.get_feature_names_out()
        except Exception:
            feature_names = [f"f{i}" for i in range(X_tr.shape[1])]

        X_train_df = pd.DataFrame(X_tr, columns=feature_names)
        X_test_df = pd.DataFrame(X_te, columns=feature_names)

        print("Shapes:")
        print("  X_train:", X_train_df.shape, " y_train:", y.shape)
        print("  X_test :", X_test_df.shape)

        return X_train_df, y, X_test_df, pre
