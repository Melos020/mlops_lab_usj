# scripts/evaluate.py

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score

from my_package.models import TitanicLogRegModel, TitanicRFModel


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate trained Titanic models on featurized data")
    p.add_argument(
        "--model",
        required=True,
        help="Path to trained model (.joblib), e.g. models/titanic_logreg.joblib",
    )
    p.add_argument(
        "--input_X",
        required=True,
        help="Path to featurized features CSV (e.g. data/processed/X_train.csv)",
    )
    p.add_argument(
        "--input_y",
        required=True,
        help="Path to targets CSV (e.g. data/processed/y_train.csv)",
    )
    p.add_argument(
        "--output",
        required=False,
        help="Optional path to save metrics JSON (e.g. reports/logreg_metrics.json)",
    )
    return p


def load_model(path: Path):
    name = path.name.lower()
    if "rf" in name:
        return TitanicRFModel.load(path)
    else:
        # default to logistic regression
        return TitanicLogRegModel.load(path)


def main() -> None:
    args = build_parser().parse_args()

    X = pd.read_csv(args.input_X).select_dtypes(include=["number"])
    y_df = pd.read_csv(args.input_y)

    if "Survived" in y_df.columns:
        y = y_df["Survived"]
    else:
        y = y_df.iloc[:, 0]

    model_path = Path(args.model)
    model = load_model(model_path)

    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)

    print(f"Evaluating model at: {model_path}")
    print(f"Accuracy: {acc:.4f}")

    if args.output:
        import json, os

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"accuracy": acc}, f, indent=2)
        print(f"Metrics saved to {out_path}")


if __name__ == "__main__":
    main()
