import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from data_processing import *
from model import *
from visualization import *
from config import *

def main():
    print("=" * 50)
    print("DOG BREED CLASSIFICATION PROJECT")
    print("=" * 50)
    
    print(f"TensorFlow version: {tf.__version__}")
    print("GPU setup will be handled during model creation...")
    
    print("\n1. Loading and processing data...")
    labels_csv = load_labels(LABELS_CSV)
    
    filenames, integer_labels, unique_breeds = create_filenames_and_labels(
        labels_csv, TRAIN_PATH, NUM_IMAGES
    )
    
    print(f"Number of unique breeds: {len(unique_breeds)}")
    print(f"Using {len(filenames)} images for training")
    
    x_train, x_val, y_train, y_val = split_data(filenames, integer_labels)
    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    
    # Calculate class weights to handle class imbalance
    class_weights = None
    if USE_CLASS_WEIGHTS:
        print("\nCalculating class weights to handle imbalanced classes...")
        class_weights_array = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = {i: weight for i, weight in enumerate(class_weights_array)}
        print(f"Class weights: {class_weights}")
    
    print("\n2. Creating data batches...")
    train_data = create_data_batches(x_train, y_train)
    val_data = create_data_batches(x_val, y_val, valid_data=True)
    
    print("\n3. Training model (Stage 1: Feature extraction)...")
    model, history = train_model(train_data, val_data, class_weights=class_weights, epochs=25)
    
    print("\n4. Fine-tuning model (Stage 2: Fine-tuning)...")
    model, fine_tune_history = fine_tune_model(model, train_data, val_data, class_weights=class_weights, epochs=35)
    
    print("\n5. Saving model...")
    model_path = save_model(model, suffix="efficientnet_finetuned")
    
    print("\n6. Making predictions...")
    predictions = model.predict(val_data, verbose=1)
    
    print("\n7. Evaluating model...")
    evaluation = model.evaluate(val_data)
    print(f"Validation accuracy: {evaluation[1]:.4f}")
    
    print("\n8. Visualizing results...")
    val_images, val_labels = unbatchify(val_data, unique_breeds)
    
    plot_predictions_grid(
        predictions, val_labels, val_images, unique_breeds, 
        start_idx=0, num_rows=2, num_cols=2
    )
    
    print("\nTraining completed successfully!")
    print(f"Model saved to: {model_path}")


def train_full_model():
    print("Training on full dataset...")
    
    labels_csv = load_labels(LABELS_CSV)
    filenames, integer_labels, unique_breeds = create_filenames_and_labels(
        labels_csv, TRAIN_PATH
    )
    
    full_data = create_data_batches(filenames, integer_labels)
    
    full_model = create_model()
    
    tensorboard = create_tensorboard_callback()
    early_stopping = create_early_stopping_callback()
    checkpoint = create_model_checkpoint_callback()
    
    full_model.fit(
        x=full_data,
        epochs=NUM_EPOCHS,
        callbacks=[tensorboard, early_stopping, checkpoint]
    )
    
    model_path = save_model(full_model, suffix="full_dataset_efficientnet_adam")
    print(f"Full model saved to: {model_path}")
    
    return full_model


def make_test_predictions(model_path):
    print("Making test predictions...")
    
    model = load_model(model_path)
    
    test_filenames = [TEST_PATH + fname for fname in os.listdir(TEST_PATH)]
    test_data = create_data_batches(test_filenames, test_data=True)
    
    test_predictions = model.predict(test_data, verbose=1)
    
    np.savetxt(DATA_PATH + "test_predictions.csv", test_predictions, delimiter=",")
    
    print("Test predictions saved!")
    return test_predictions


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "full":
            train_full_model()
        elif sys.argv[1] == "test":
            if len(sys.argv) > 2:
                make_test_predictions(sys.argv[2])
            else:
                print("Please provide model path for testing")
        else:
            main()
    else:
        main()
