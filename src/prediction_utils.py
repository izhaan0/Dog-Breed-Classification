"""
Prediction utility module for Dog Breed Classification project
"""

import os
import numpy as np
import pandas as pd
from data_processing import create_data_batches
from visualization import get_pred_label, plot_custom_predictions
from model import load_model
from config import *


def predict_custom_images(model_path, custom_image_paths, unique_breeds):
    """Make predictions on custom images"""
    print("Making predictions on custom images...")
    
    # Load model
    model = load_model(model_path)
    
    # Create data batches for custom images
    custom_data = create_data_batches(custom_image_paths, test_data=True)
    
    # Make predictions
    custom_predictions = model.predict(custom_data)
    
    # Get prediction labels
    custom_pred_labels = [
        get_pred_label(custom_predictions[i], unique_breeds) 
        for i in range(len(custom_predictions))
    ]
    
    # Get images for visualization
    custom_images = []
    for image in custom_data.unbatch().as_numpy_iterator():
        custom_images.append(image)
    
    # Plot results
    plot_custom_predictions(custom_images, custom_pred_labels)
    
    return custom_predictions, custom_pred_labels


def create_submission_file(test_predictions, unique_breeds, test_path, output_path):
    """Create submission file for Kaggle competition"""
    print("Creating submission file...")
    
    # Create DataFrame
    preds_df = pd.DataFrame(columns=["id"] + list(unique_breeds))
    
    # Add test IDs
    test_ids = [os.path.splitext(path)[0] for path in os.listdir(test_path)]
    preds_df["id"] = test_ids
    
    # Add predictions
    preds_df[list(unique_breeds)] = test_predictions
    
    # Save to CSV
    preds_df.to_csv(output_path, index=False)
    print(f"Submission file saved to: {output_path}")
    
    return preds_df


def evaluate_predictions(predictions, true_labels, unique_breeds):
    """Evaluate prediction accuracy"""
    pred_labels = [get_pred_label(pred, unique_breeds) for pred in predictions]
    
    correct = sum(1 for pred, true in zip(pred_labels, true_labels) if pred == true)
    accuracy = correct / len(true_labels)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Correct predictions: {correct}/{len(true_labels)}")
    
    return accuracy


def get_top_n_predictions(predictions, unique_breeds, n=5):
    """Get top N predictions for each sample"""
    results = []
    
    for pred in predictions:
        # Get top N indices
        top_indices = pred.argsort()[-n:][::-1]
        top_labels = unique_breeds[top_indices]
        top_probs = pred[top_indices]
        
        results.append(list(zip(top_labels, top_probs)))
    
    return results
