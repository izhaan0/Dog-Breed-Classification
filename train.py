import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import json
from PIL import Image
import glob

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001

def load_data():
    """Load and prepare the dataset"""
    print("Loading data...")
    
    # Load labels
    labels_df = pd.read_csv('data/labels.csv')
    print(f"Loaded {len(labels_df)} labels")
    
    # Get all image paths
    image_paths = []
    labels = []
    
    for _, row in labels_df.iterrows():
        img_path = f"data/train/{row['id']}.jpg"
        if os.path.exists(img_path):
            image_paths.append(img_path)
            labels.append(row['breed'])
    
    print(f"Found {len(image_paths)} images")
    print(f"Number of unique breeds: {len(set(labels))}")
    
    return image_paths, labels

def preprocess_image(image_path, label):
    """Preprocess image for training"""
    # Read image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    
    # Resize and normalize
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = tf.cast(image, tf.float32) / 255.0
    
    return image, label

def create_data_augmentation():
    """Create data augmentation pipeline"""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.RandomBrightness(0.1),
    ])

def create_model(num_classes):
    """Create MobileNetV2 model with transfer learning"""
    print("Creating model...")
    
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Add custom classification head
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = create_data_augmentation()(inputs)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    return model

def main():
    """Main training function"""
    print("Starting dog breed classification training...")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    
    # Load data
    image_paths, labels = load_data()
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    
    # Save class indices for later use
    class_indices = {i: class_name for i, class_name in enumerate(label_encoder.classes_)}
    with open('models/class_indices.json', 'w') as f:
        json.dump(class_indices, f, indent=2)
    
    print(f"Class indices saved to models/class_indices.json")
    print(f"Number of classes: {num_classes}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Create model
    model = create_model(num_classes)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'models/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            verbose=1,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Train model
    print("Starting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save('models/dog_breed_model.h5')
    print("Model saved to models/dog_breed_model.h5")
    
    # Save training history
    history_dict = {
        'accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'],
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss']
    }
    
    with open('models/training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print("Training history saved to models/training_history.json")
    
    # Final evaluation
    print("\nFinal model evaluation:")
    val_loss, val_accuracy = model.evaluate(val_dataset, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    print("\nTraining completed successfully!")
    print(f"Model files saved in 'models/' directory")

if __name__ == "__main__":
    main()
