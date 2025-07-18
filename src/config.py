DATA_PATH = "data/"
TRAIN_PATH = DATA_PATH + "train/"
TEST_PATH = DATA_PATH + "test/"
LABELS_CSV = DATA_PATH + "labels.csv"
MODELS_PATH = "models/"
LOGS_PATH = "logs/"

# Increased image size for better feature extraction
IMG_SIZE = 300
INPUT_SHAPE = (None, IMG_SIZE, IMG_SIZE, 3)
BATCH_SIZE = 32  # Reduced batch size for better generalization
NUM_EPOCHS = 100
NUM_IMAGES = None  # Use all available images

# Removed TF Hub model URL as we're using EfficientNet directly

TRAIN_TEST_SPLIT = 0.15  # Reduced validation split to have more training data
RANDOM_STATE = 42
EARLY_STOPPING_PATIENCE = 10  # Increased patience for better convergence

# Learning rate settings
INITIAL_LR = 1e-4
MIN_LR = 1e-6
LR_FACTOR = 0.2

# Class balancing
USE_CLASS_WEIGHTS = True
