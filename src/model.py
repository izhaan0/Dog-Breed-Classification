import os
import datetime
import tensorflow as tf
import tensorflow_hub as hub
from config import *

def setup_gpu():
    """Configure GPU settings and return device to use"""
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            print(f"✓ GPU available: {len(gpus)} GPU(s) detected")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")
            
            # Check if ROCm is available (for AMD GPUs)
            if hasattr(tf.test, 'is_built_with_rocm'):
                print(f"  ROCm support: {tf.test.is_built_with_rocm()}")
            
            return '/GPU:0'
            
        except RuntimeError as e:
            print(f"✗ GPU setup failed: {e}")
            print("  Falling back to CPU")
            return '/CPU:0'
    else:
        print("✗ No GPU detected - using CPU")
        print("  For faster training, consider:")
        print("  1. Running setup_gpu.py for AMD GPU setup")
        print("  2. Using Google Colab with GPU runtime")
        return '/CPU:0'

def create_model(input_shape=INPUT_SHAPE, output_shape=120):
    print("Building improved EfficientNetB3 model...")
    
    # Setup GPU and get device
    device = setup_gpu()
    print(f"Using device: {device}")
    
    with tf.device(device):
        inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        
        # We'll use minimal augmentation here since we're already doing it in the data pipeline
        # This is just for additional randomness during training
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
            tf.keras.layers.RandomContrast(0.1),
        ])
        
        x = augmentation(inputs)
        
        # Using a more powerful EfficientNetB3 model
        base_model = tf.keras.applications.EfficientNetB3(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )
        
        base_model.trainable = False
        x = base_model(x, training=False)
        
        # Enhanced feature extraction head
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        # First dense block with residual connection
        block_input = x
        x = tf.keras.layers.Dense(1536, activation=None)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(1536, activation=None)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        # Add residual connection if dimensions match
        if block_input.shape[-1] == 1536:
            x = tf.keras.layers.add([x, block_input])
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        
        # Second dense layer
        x = tf.keras.layers.Dense(768, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Output layer with label smoothing for better generalization
        outputs = tf.keras.layers.Dense(output_shape, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        # Use a lower learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_accuracy')]
        )
        
    return model


def create_tensorboard_callback(log_dir=LOGS_PATH):
    logdir = os.path.join(
        log_dir,
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    return tf.keras.callbacks.TensorBoard(logdir)

def create_early_stopping_callback(monitor='val_accuracy', patience=EARLY_STOPPING_PATIENCE):
    return tf.keras.callbacks.EarlyStopping(
        monitor=monitor, 
        patience=patience,
        restore_best_weights=True
    )

def create_reduce_lr_callback():
    return tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=LR_FACTOR,
        patience=3,
        min_lr=MIN_LR,
        verbose=1
    )

def create_model_checkpoint_callback(model_dir=MODELS_PATH):
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = os.path.join(model_dir, f"best_model_{timestamp}.keras")
    
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        verbose=1
    )

def fine_tune_model(model, train_data, val_data, class_weights=None, epochs=30):
    print("Starting fine-tuning with progressive layer unfreezing...")
    
    # Get the base model (EfficientNetB3)
    base_model = model.layers[3]
    
    # First stage: Train only the top layers (the layers that we added on top of EfficientNet)
    print("Stage 1: Training only top layers...")
    base_model.trainable = False
    
    # Compile with a lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=MIN_LR * 10),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_accuracy')]
    )
    
    # Create callbacks
    tensorboard = create_tensorboard_callback()
    early_stopping = create_early_stopping_callback(patience=7)
    reduce_lr = create_reduce_lr_callback()
    checkpoint = create_model_checkpoint_callback()
    
    # Train top layers
    print("Fine-tuning top layers...")
    history1 = model.fit(
        train_data,
        epochs=10,
        validation_data=val_data,
        class_weight=class_weights,
        callbacks=[tensorboard, early_stopping, reduce_lr, checkpoint]
    )
    
    # Second stage: Fine-tune the last few blocks of EfficientNetB3
    print("Stage 2: Fine-tuning last few blocks...")
    base_model.trainable = True
    
    # Freeze all the layers except the last 50
    fine_tune_at = -50  # Unfreeze the last 50 layers
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    # Print trainable layers
    print(f"Total layers in base model: {len(base_model.layers)}")
    print(f"Number of trainable layers: {sum(1 for layer in base_model.layers if layer.trainable)}")
    
    # Recompile with an even lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=MIN_LR * 5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_accuracy')]
    )
    
    # Train with fine-tuning
    print("Fine-tuning last blocks...")
    history2 = model.fit(
        train_data,
        epochs=epochs,
        initial_epoch=history1.epoch[-1] + 1 if history1.epoch else 0,
        validation_data=val_data,
        class_weight=class_weights,
        callbacks=[tensorboard, early_stopping, reduce_lr, checkpoint]
    )
    
    # Combine histories
    combined_history = {}
    for k in history1.history.keys():
        if k in history2.history:
            combined_history[k] = history1.history[k] + history2.history[k]
    
    return model, combined_history


def train_model(train_data, val_data, class_weights=None, epochs=NUM_EPOCHS):
    model = create_model()
    
    # Print model summary
    model.summary()
    
    # Create callbacks
    tensorboard = create_tensorboard_callback()
    early_stopping = create_early_stopping_callback()
    checkpoint = create_model_checkpoint_callback()
    reduce_lr = create_reduce_lr_callback()
    
    # Add learning rate scheduler for better convergence
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: INITIAL_LR * (0.9 ** epoch)
    )
    
    # Train the model with class weights if provided
    print(f"Training with{'out' if class_weights is None else ''} class weights...")
    history = model.fit(
        x=train_data,
        epochs=epochs,
        validation_data=val_data,
        validation_freq=1,
        class_weight=class_weights,
        callbacks=[tensorboard, early_stopping, checkpoint, reduce_lr, lr_scheduler]
    )
    
    return model, history

def save_model(model, suffix=None, model_dir=MODELS_PATH):
    os.makedirs(model_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(model_dir, f"{timestamp}_{suffix}.keras")
    
    print(f"Saving model to: {model_path}")
    model.save(model_path)
    return model_path

def load_model(model_path):
    print(f"Loading saved model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model
