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
BATCH_SIZE = 15
# Batch size is the bottleneck. InceptionResnetV2 is a wide and deep network,
# so itself is already heavy for home GPUs, even more considering during
# training higher TRAIN_DEPTH. Additionally, we have to use not so small images
# so we don't lose too much information. To make it all fit in the memory, each
# step has to take just a few images into consideration, in other words, small
# batch size. As a consequence, the model has more variance and takes longer to
# start converging.

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

TRAIN_DEPTH = 2

# For dropout
KEEP_PROB = 0.8

# 299 is the standard inception size.
IMG_WIDTH = 299
IMG_HEIGHT = 299

#  Model Classes
CLASSES = 3
NUM_READERS = 4

# Indexes and strings
TRAIN = 'train'
VALID = 'validation'
