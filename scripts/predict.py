import argparse
import pandas as pd
import os
import pickle
 
def build_parser():
    p = argparse.ArgumentParser(description="Make predictions using trained Titanic model")
    p.add_argument("--model", required=True, help="Path to trained model pickle file")
    p.add_argument("--input", required=True, help="Path to feature CSV (no labels)")
    p.add_argument("--output", required=True, help="Path to save predictions CSV")
    return p
 
def main():
    args = build_parser().parse_args()
   
    # Load features
    df = pd.read_csv(args.input)
   
    # Load trained model
    with open(args.model, 'rb') as f:
        model = pickle.load(f)
   
    # Predict
    y_pred = model.predict(df)
   
    # Save predictions
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_pred = pd.DataFrame({"Prediction": y_pred})
    df_pred.to_csv(args.output, index=False)
   
    print(f"Predictions saved to {args.output}")
 
if __name__ == "__main__":
    main()