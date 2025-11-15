set -e

echo "=== Preprocessing raw data ==="
python scripts/preprocess.py \
    --input data/raw/titanic.csv \
    --output data/processed

echo "=== Featurizing training data ==="
python scripts/featurize.py \
    --input data/processed/X_train_raw.csv \
    --output data/processed/X_train.csv

echo "=== Training Logistic Regression ==="
python scripts/train.py \
    --input_X data/processed/X_train.csv \
    --input_y data/processed/y_train.csv \
    --output models/titanic_logreg.joblib

echo "=== Training Random Forest ==="
python scripts/train.py \
    --input_X data/processed/X_train.csv \
    --input_y data/processed/y_train.csv \
    --output models/titanic_rf.joblib \
    --backend rf

echo "=== Evaluating Logistic Regression ==="
python scripts/evaluate.py \
    --model models/titanic_logreg.joblib \
    --input data/processed/train_with_labels.csv \
    --output reports/logreg_metrics.json

echo "=== Evaluating Random Forest ==="
python scripts/evaluate.py \
    --model models/titanic_rf.joblib \
    --input data/processed/train_with_labels.csv \
    --output reports/rf_metrics.json

echo "=== Pipeline complete ==="
