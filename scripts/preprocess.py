"""
Data preprocessing script for Titanic survival prediction.
Now delegates the actual logic to the TitanicPreprocessor class.
"""

import argparse
from pathlib import Path

import pandas as pd

from my_package.preprocessing.preprocessor import TitanicPreprocessor


def main():
    parser = argparse.ArgumentParser(description="Preprocess Titanic dataset")
    parser.add_argument(
        "--train_path", type=str, required=True, help="Path to training CSV file"
    )
    parser.add_argument(
        "--test_path", type=str, required=True, help="Path to test CSV file"
    )
    parser.add_argument(
        "--output_train",
        type=str,
        required=True,
        help="Output path for preprocessed training data",
    )
    parser.add_argument(
        "--output_test",
        type=str,
        required=True,
        help="Output path for preprocessed test data",
    )

    args = parser.parse_args()

    # Create output directories if they don't exist
    Path(args.output_train).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_test).parent.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_raw = pd.read_csv(args.train_path)
    test_raw = pd.read_csv(args.test_path)
    print(f"Loaded train: {train_raw.shape}, test: {test_raw.shape}")

    # Use the class instead of local functions
    preprocessor = TitanicPreprocessor()

    print("Running TitanicPreprocessor...")
    train_processed, test_processed = preprocessor.process(train_raw, test_raw)

    print("Saving preprocessed data...")
    train_processed.to_csv(args.output_train, index=False)
    test_processed.to_csv(args.output_test, index=False)

    print(f"Preprocessed train saved to: {args.output_train}")
    print(f"Preprocessed test saved to: {args.output_test}")
    print(f"Final train shape: {train_processed.shape}")
    print(f"Final test shape: {test_processed.shape}")


if __name__ == "__main__":
    main()
