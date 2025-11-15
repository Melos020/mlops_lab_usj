# scripts/train.py

import argparse
from pathlib import Path

import pandas as pd

from my_package.models.logreg_model import TitanicLogRegModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Titanic logistic regression model on featurized data")
    parser.add_argument(
        "--input_X",
        required=True,
        help="Path to featurized training features CSV (e.g. data/processed/X_train.csv)",
    )
    parser.add_argument(
        "--input_y",
        required=True,
        help="Path to training targets CSV (e.g. data/processed/y_train.csv)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save trained model (e.g. models/titanic_logreg.joblib)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    X = pd.read_csv(args.input_X)
    y_df = pd.read_csv(args.input_y)

    # NEW: keep only numeric columns (drop Name, Ticket, Cabin, etc.)
    X = X.select_dtypes(include=["number"])

    if "Survived" in y_df.columns:
        y = y_df["Survived"]
    else:
        y = y_df.iloc[:, 0]

    print(f"Loaded X shape: {X.shape}, y shape: {y.shape}")

    model = TitanicLogRegModel()
    print("Fitting TitanicLogRegModel on featurized data...")
    model.fit(X, y)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    print(f"Trained model saved to {out_path}")



if __name__ == "__main__":
    main()
