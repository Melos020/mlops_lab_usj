import argparse
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
 
def build_parser():
    p = argparse.ArgumentParser(description="Train Titanic baseline model")
    p.add_argument("--input", required=True, help="Path to feature CSV")
    p.add_argument("--output", required=True, help="Path to save trained model")
    return p
 
def main():
    args = build_parser().parse_args()
   
    df = pd.read_csv(args.input)
   
    if 'Survived' not in df.columns:
        raise ValueError("Input CSV must contain 'Survived' column")
   
    X = df.drop(columns=['Survived'])
    y = df['Survived']
   
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
   
    # Preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('num', MinMaxScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])
   
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
   
    pipeline.fit(X, y)
   
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
   
    # Save model
    with open(args.output, 'wb') as f:
        pickle.dump(pipeline, f)
   
    print(f"Trained model saved to {args.output}")
 
if __name__ == "__main__":
    main()