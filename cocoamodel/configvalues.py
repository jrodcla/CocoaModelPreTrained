
# TODO Document usage of all these constants.

# Basic Configuration
CHECKPOINT_FILE = './inception/inception_resnet_v2_2016_08_30.ckpt'
DATASET_NAME = 'plants'
DATASET_DIR = './dataset'
TRAIN_DIR = './train_runs/train'
EVAL_DIR = './eval_runs/eval'
LOG_DIR = './log'
LOG_FREQ = 10


# Process Configuration
NUM_THREADS = 4

# Training information

NUM_EPOCHS = 2500  # the number of epochs to train
BATCH_SIZE = 20

EXAMPLES_PER_EVAL = 500

MIN_FRACTION_QUEUE = 0.4  # Minimum fraction of examples in the queue.
# EXAMPLES_PER_EPOCH = 1000 # Number of examples per epoch during training.
QUEUE_MIN = BATCH_SIZE
QUEUE_CAPAC = 4 * BATCH_SIZE

# Number of epochs before decaying the learning rate.
EPOCHS_BEFORE_DECAY = 250
# Initial learning rate. Up than 0,005 it overshoots.
INITIAL_LEARN_RATE = 0.001
LEARN_DECAY_FACTOR = 0.40  # Learning rate decay factor.

TRAIN_DEPTH = 0

# For dropout
KEEP_PROB = 0.8

# 299 is the standard inception size.
IMG_WIDTH = 299
IMG_HEIGHT = 299

#  Model Classes
CLASSES = 2
NUM_READERS = 4

# Indexes and strings
TRAIN = 'train'
VALID = 'validation'
