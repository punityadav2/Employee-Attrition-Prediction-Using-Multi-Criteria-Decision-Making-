"""
Prediction script for HR Analytics project
Usage: python predict.py --model models/best_model.pkl --data data/raw/aug_test.csv
"""

import argparse
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model_training import load_model
from src.preprocessing import handle_missing_values, encode_categorical
from src.feature_engineering import create_all_features
import config


def load_and_preprocess_data(data_path, encoders=None):
    """
    Load and preprocess data for prediction
    
    Args:
        data_path (str): Path to data file
        encoders (dict): Pre-fitted encoders
        
    Returns:
        tuple: (processed dataframe, enrollee_ids)
    """
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} records")
    
    # Save enrollee_ids
    enrollee_ids = df['enrollee_id'].copy()
    
    # Preprocess
    print("\nPreprocessing data...")
    df = handle_missing_values(df, strategy='mode')
    df = create_all_features(df)
    df, _ = encode_categorical(df, fit=False, encoders=encoders)
    
    # Drop ID column
    if 'enrollee_id' in df.columns:
        df = df.drop(columns=['enrollee_id'])
    
    # Drop target if present
    if config.TARGET_COLUMN in df.columns:
        df = df.drop(columns=[config.TARGET_COLUMN])
    
    print(f"✓ Preprocessing complete. Features: {df.shape[1]}")
    
    return df, enrollee_ids


def make_predictions(model, X):
    """
    Make predictions using trained model
    
    Args:
        model: Trained model
        X (pd.DataFrame): Features
        
    Returns:
        tuple: (predictions, probabilities)
    """
    print("\nMaking predictions...")
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
    
    print(f"✓ Predictions complete")
    print(f"  Predicted 'Looking for job change': {predictions.sum()} ({predictions.sum()/len(predictions)*100:.1f}%)")
    print(f"  Predicted 'Not looking': {(1-predictions).sum()} ({(1-predictions).sum()/len(predictions)*100:.1f}%)")
    
    return predictions, probabilities


def save_predictions(enrollee_ids, predictions, probabilities, output_path):
    """
    Save predictions to CSV file
    
    Args:
        enrollee_ids (pd.Series): Enrollee IDs
        predictions (array): Predicted labels
        probabilities (array): Predicted probabilities
        output_path (str): Path to save predictions
    """
    results = pd.DataFrame({
        'enrollee_id': enrollee_ids,
        'target': predictions
    })
    
    if probabilities is not None:
        results['probability'] = probabilities
    
    results.to_csv(output_path, index=False)
    print(f"\n✓ Predictions saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Make predictions using trained HR Analytics model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model file')
    parser.add_argument('--data', type=str, required=True, help='Path to data file for prediction')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Path to save predictions')
    parser.add_argument('--encoders', type=str, default=None, help='Path to saved encoders (optional)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("  HR ANALYTICS - PREDICTION SCRIPT")
    print("="*70)
    
    # Load model
    model = load_model(args.model, model_dir='')
    if model is None:
        print("✗ Error: Could not load model")
        return
    
    # Load encoders if provided
    encoders = None
    if args.encoders:
        import joblib
        encoders = joblib.load(args.encoders)
        print(f"✓ Loaded encoders from {args.encoders}")
    
    # Load and preprocess data
    X, enrollee_ids = load_and_preprocess_data(args.data, encoders)
    
    # Make predictions
    predictions, probabilities = make_predictions(model, X)
    
    # Save predictions
    save_predictions(enrollee_ids, predictions, probabilities, args.output)
    
    print("\n" + "="*70)
    print("  PREDICTION COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
