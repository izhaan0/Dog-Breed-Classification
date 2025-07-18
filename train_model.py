import os
import json
import numpy as np
import pandas as pd
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


def organize_data_by_breed(source_dir, labels_file, organized_dir):
    """Organize images into breed subdirectories"""
    print("Organizing data by breed...")
    
    # Read labels
    labels_df = pd.read_csv(labels_file)
    
    # Create organized directory structure
    if os.path.exists(organized_dir):
        shutil.rmtree(organized_dir)
    os.makedirs(organized_dir)
    
    # Create breed subdirectories and move images
    for _, row in labels_df.iterrows():
        image_id = row['id']
        breed = row['breed']
        
        # Create breed directory if it doesn't exist
        breed_dir = os.path.join(organized_dir, breed)
        if not os.path.exists(breed_dir):
            os.makedirs(breed_dir)
        
        # Copy image to breed directory
        src_path = os.path.join(source_dir, f"{image_id}.jpg")
        dst_path = os.path.join(breed_dir, f"{image_id}.jpg")
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"Warning: Image {src_path} not found")
    
    print(f"Data organized into {len(labels_df['breed'].unique())} breed directories")


def load_data(data_dir, img_size=(224, 224)):
    print("Loading data...")
    datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    return train_generator, validation_generator


def build_model(num_classes):
    print("Building model...")
    base_model = MobileNetV2(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, train_generator, validation_generator):
    print("Training model...")
    model.fit(train_generator, epochs=3, validation_data=validation_generator)


def save_model(model, class_indices):
    print("Saving model...")
    model.save('models/model.h5')
    with open('models/class_indices.json', 'w') as f:
        json.dump(class_indices, f)


def main():
    # Organize data by breed first
    source_dir = 'data/train'
    labels_file = 'data/labels.csv'
    organized_dir = 'data/dog_images'
    
    organize_data_by_breed(source_dir, labels_file, organized_dir)
    
    # Load data and train model
    train_generator, validation_generator = load_data(organized_dir)
    model = build_model(num_classes=len(train_generator.class_indices))
    train_model(model, train_generator, validation_generator)
    save_model(model, train_generator.class_indices)


if __name__ == "__main__":
    main()
