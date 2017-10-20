
##TODO Document usage of all these constants.

##################### Basic Configuration
CHECKPOINT_FILE = './inception/inception_resnet_v2_2016_08_30.ckpt'
DATASET_NAME    = 'plants'
DATASET_DIR     = './dataset'
TRAIN_DIR       = './train_runs/train'
EVAL_DIR        = './eval_runs/eval'
LOG_DIR         = './log'
LOG_FREQ        = 10


##################### Process Configuration
NUM_THREADS = 4

##################### Training information

NUM_EPOCHS = 2500 # the number of epochs to train
BATCH_SIZE = 20

EXAMPLES_PER_EVAL = 500

MIN_FRACTION_QUEUE = 0.4  # Minimum fraction of examples in the queue.
#EXAMPLES_PER_EPOCH = 1000 # Number of examples per epoch during training.
#BATCHES_PER_EPOCH  = EXAMPLES_PER_EPOCH / BATCH_SIZE
QUEUE_MIN          = BATCH_SIZE
QUEUE_CAPAC        = 4 * BATCH_SIZE

EPOCHS_BEFORE_DECAY = 100 # Number of epochs before decaying the learning rate.
#DECAY_STEPS         = int((EXAMPLES_PER_EPOCH / BATCH_SIZE) * EPOCHS_BEFORE_DECAY)
INITIAL_LEARN_RATE  = 0.001 #  Initial learning rate. Up than 0,005 it overshoots.
LEARN_DECAY_FACTOR  = 0.40 #  Learning rate decay factor.
MOVING_AVG_DECAY    = 0.999 # Decay for the moving average.

TRAIN_DEPTH = 2

##################### Layers

# For conv 1 :
KERNEL_SIZE     = 5
OUTPUT_CHANNEL  = 64

# For pool 1 :
WINDOW_SIZE     = 3

# For normalization :
BIAS  = 1.0
ALPHA = 0.001 / 9.0
BETA  = 0.75
DEPTH = 4

# For dropout
KEEP_PROB = 0.8

##################### Dataset information

# 299 is the standard inception size.
IMG_WIDTH   = 299
IMG_HEIGHT  = 299
IMG_CHANNEL = 3

CLASSES     = 2 #  Model Classes
NUM_READERS = 4

# Image distortion params
MAX_DELTA   = 63
LOWER_CONTR = 0.2
UPPER_CONTR = 1.8

##################### Indexes and strings

TRAIN = 'train'
VALID = 'validation'
