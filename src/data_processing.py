import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from config import *

def load_labels(csv_path):
    labels_csv = pd.read_csv(csv_path)
    print(f"Loaded {len(labels_csv)} labels")
    print(labels_csv.describe())
    return labels_csv

def create_filenames_and_labels(labels_csv, train_path, num_images=None):
    filenames = [train_path + fname + ".jpg" for fname in labels_csv["id"]]
    labels = labels_csv["breed"].to_numpy()
    
    if num_images:
        filenames = filenames[:num_images]
        labels = labels[:num_images]
    
    unique_breeds = np.unique(labels)
    label_to_int = {label: idx for idx, label in enumerate(unique_breeds)}
    integer_labels = [label_to_int[label] for label in labels]
    
    return filenames, integer_labels, unique_breeds

def process_image(image_path, img_size=IMG_SIZE):
    """Enhanced image processing with better resizing and normalization"""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    
    # Better resizing with preserve_aspect_ratio
    image = tf.image.resize(image, size=[img_size, img_size], 
                           preserve_aspect_ratio=True,
                           antialias=True)
    
    # Handle potential dimension issues from preserve_aspect_ratio
    image = tf.image.resize_with_crop_or_pad(image, img_size, img_size)
    
    # Convert to float and normalize to 0-1 range
    image = tf.cast(image, tf.float32) / 255.0
    
    return image

def get_image_label(image_path, label):
    image = process_image(image_path)
    return image, label

def apply_augmentation(image, label):
    """Apply additional augmentation to training images"""
    # Random color adjustments
    image = tf.image.random_saturation(image, 0.7, 1.3)
    image = tf.image.random_hue(image, 0.05)
    
    # Random flips
    image = tf.image.random_flip_left_right(image)
    
    # Safe random crop and resize
    # First ensure the image has the right dimensions
    image_shape = tf.shape(image)
    crop_size = tf.cast(tf.cast(IMG_SIZE, tf.float32) * 0.9, tf.int32)
    
    # Only crop if the image is large enough
    if (image_shape[0] >= crop_size) and (image_shape[1] >= crop_size):
        image = tf.image.random_crop(image, [crop_size, crop_size, 3])
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    
    return image, label

def create_data_batches(x, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
    if test_data:
        print("Creating test data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x)))
        data_batch = data.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
        data_batch = data_batch.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return data_batch
    
    elif valid_data:
        print("Creating validation data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x), tf.constant(y)))
        data_batch = data.map(get_image_label, num_parallel_calls=tf.data.AUTOTUNE)
        data_batch = data_batch.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return data_batch
    
    else:
        print("Creating training data batches with enhanced augmentation...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x), tf.constant(y)))
        # Use a larger shuffle buffer for better randomization
        data = data.shuffle(buffer_size=min(len(x), 10000))
        # Load and process images
        data = data.map(get_image_label, num_parallel_calls=tf.data.AUTOTUNE)
        # Apply augmentation
        data = data.map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
        # Use repeat to create more variations during training
        data = data.repeat(1)
        # Batch and prefetch
        data_batch = data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return data_batch

def split_data(filenames, labels, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_STATE):
    x_train, x_val, y_train, y_val = train_test_split(
        filenames, labels, test_size=test_size, random_state=random_state
    )
    return x_train, x_val, y_train, y_val

def unbatchify(data, unique_breeds):
    images = []
    labels = []
    
    for image, label in data.unbatch().as_numpy_iterator():
        images.append(image)
        labels.append(unique_breeds[label])
    
    return images, labels
