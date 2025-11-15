import argparse
from pathlib import Path

import joblib
import pandas as pd

from my_package.features.features_computer import TitanicFeaturesComputer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Featurize Titanic data after preprocessing")
    p.add_argument("--train_path", type=str, required=True, help="Path to preprocessed training CSV")
    p.add_argument("--test_path", type=str, required=True, help="Path to preprocessed test CSV")
    p.add_argument("--out_train_X", type=str, required=True, help="Where to save transformed X_train (CSV)")
    p.add_argument("--out_train_y", type=str, required=True, help="Where to save y_train (CSV)")
    p.add_argument("--out_test_X", type=str, required=True, help="Where to save transformed X_test (CSV)")
    p.add_argument(
        "--preprocessor_path",
        type=str,
        required=True,
        help="Where to save fitted preprocessor (joblib)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    # Ensure output dirs exist
    for p_out in [args.out_train_X, args.out_train_y, args.out_test_X, args.preprocessor_path]:
        Path(p_out).parent.mkdir(parents=True, exist_ok=True)

    print("Loading preprocessed data...")
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)

    # Use the class
    feats = TitanicFeaturesComputer(test_df=test_df)
    X_train_df, y_train, X_test_df, preprocessor = feats.compute(train_df)

    # Save outputs
    print("Saving featurized data and preprocessor...")
    X_train_df.to_csv(args.out_train_X, index=False)
    y_train.to_csv(args.out_train_y, index=False, header=True)
    X_test_df.to_csv(args.out_test_X, index=False)
    joblib.dump(preprocessor, args.preprocessor_path)

    print(f"Saved X_train to: {args.out_train_X}")
    print(f"Saved y_train to: {args.out_train_y}")
    print(f"Saved X_test to:  {args.out_test_X}")
    print(f"Saved preprocessor to: {args.preprocessor_path}")
    print("Featurization complete.")


if __name__ == "__main__":
    main()
