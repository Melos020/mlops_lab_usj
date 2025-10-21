import argparse
import pandas as pd
import os
import pickle
import json
from sklearn.metrics import accuracy_score
 
def build_parser():
    p = argparse.ArgumentParser(description="Evaluate trained Titanic model")
    p.add_argument("--model", required=True, help="Path to trained model pickle file")
    p.add_argument("--input", required=True, help="Path to feature CSV with labels")
    p.add_argument("--output", required=False, help="Path to save metrics JSON (optional)")
    return p
 
def main():
    args = build_parser().parse_args()
   
    # Load feature dataset
    df = pd.read_csv(args.input)
   
    if 'Survived' not in df.columns:
        raise ValueError("Input CSV must contain 'Survived' column")
   
    X = df.drop(columns=['Survived'])
    y = df['Survived']
   
    # Load trained model
    with open(args.model, 'rb') as f:
        model = pickle.load(f)
   
    # Predict
    y_pred = model.predict(X)
   
    # Compute accuracy
    accuracy = accuracy_score(y, y_pred)
   
    print(f"Model accuracy: {accuracy:.4f}")
   
    # Optionally save metrics to JSON
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        metrics = {"accuracy": accuracy}
        with open(args.output, 'w') as f:
            json.dump(metrics, f)
        print(f"Metrics saved to {args.output}")
 
if __name__ == "__main__":
    main()