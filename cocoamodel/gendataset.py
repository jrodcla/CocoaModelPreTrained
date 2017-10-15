import os
import tensorflow as tf
import inception.inception_preprocessing as inception

# Custom imports
from cocoamodel.configvalues import *

slim = tf.contrib.slim

##################### Dictionaries for the dataset


def get_label_dict():
    """Creates a dictiorary relating each label to their string name."""
    # Find labels file and reads from it.
    labels_file = DATASET_DIR + '/labels.txt'
    labels_to_name = {}
    with open(labels_file, 'r') as labels:
        for line in labels:
            label, string_name = line.split(':')
            string_name = string_name[:-1] #Remove newline
            labels_to_name[int(label)] = string_name
    return labels_to_name


def get_itens_descr():
    """Create a description of your dictionary itens."""
    return {
    'image': 'A 3-channel RGB plants image that contain cocoa pods or not.',
    'label': 'Binary label stating 0:cocoa, 1:others'
    }


##################### Dictionaries for the decoder

def get_keys_to_features():
    return {'image/encoded': tf.FixedLenFeature((),
                                                tf.string,
                                                default_value=''),
            'image/format': tf.FixedLenFeature((),
                                               tf.string,
                                               default_value='jpg'),
            'image/class/label': tf.FixedLenFeature([],
                                                    tf.int64,
                                                    default_value=tf.zeros(
                                                        [], dtype=tf.int64))
           }

def get_itens_to_handlers():
    return {'image': slim.tfexample_decoder.Image(),
            'label': slim.tfexample_decoder.Tensor('image/class/label')
           }

##################### Useful functions

def count_samples(dt_type=TRAIN):
    """Return total of samples found in files from pattern."""
    file_pattern = DATASET_NAME + '_' + dt_type
    num_samples = 0
    for data_file in os.listdir(DATASET_DIR):
        if data_file.startswith(file_pattern):
            for record in tf.python_io.tf_record_iterator(
                    os.path.join(DATASET_DIR, data_file)):
                num_samples += 1
    return num_samples

def from_tfrecord(dt_type=TRAIN):
    """Prepare dataset to be used in the network.

    Generate dataset object with all its attributes from a tfrecord file.

    Params: TODO
    Returns: TODO
    """
    path_pattern = os.path.join(DATASET_DIR,
                                DATASET_NAME + '_' + dt_type + '_*.tfrecord')

    reader = tf.TFRecordReader
    decoder = slim.tfexample_decoder.TFExampleDecoder(get_keys_to_features(),
                                                      get_itens_to_handlers())

    dataset = slim.dataset.Dataset(data_sources = path_pattern,
                                   decoder = decoder,
                                   reader = reader,
                                   num_readers = NUM_READERS,
                                   num_samples = count_samples(dt_type),
                                   num_classes = CLASSES,
                                   labels_to_name = get_label_dict(),
                                   items_to_descriptions = get_itens_descr())

    return dataset

def _load_batch(raw_image, raw_label, batch_size=BATCH_SIZE, shuffle=True):
    # Batches could use the flag allow_smaller_final_batch, but then operations
    # requiring fixed batch size would fail.
    if shuffle:
        images, labels = tf.train.shuffle_batch([raw_image, raw_label],
                                                batch_size=batch_size,
                                                num_threads=NUM_THREADS,
                                                capacity=QUEUE_CAPAC,
                                                min_after_dequeue=QUEUE_MIN)
    else:
        images, labels = tf.train.batch([raw_image, raw_label],
                                        batch_size=batch_size,
                                        num_threads=NUM_THREADS,
                                        capacity=QUEUE_CAPAC)

    return images, tf.reshape(labels, [batch_size])

def get_batch_input(dt_type=TRAIN, no_pre_process=False):
    ##TODO decrease examples per epoch in eval mode.

    input_dataset = from_tfrecord(dt_type)

    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        input_dataset,
        common_queue_capacity = QUEUE_CAPAC,
        common_queue_min = QUEUE_MIN)

    image_data, label_data = data_provider.get(['image', 'label'])

    #TODO Do I save some memory casting to int32?
    #label_processed = tf.cast(label_data, tf.int32)

    if (no_pre_process):
        image_data = tf.expand_dims(image_data, 0)
        image_data = tf.image.resize_nearest_neighbor(image_data, 
                                                      [IMG_HEIGHT,
                                                      IMG_WIDTH])
        image_data = tf.squeeze(image_data)
    else:
        image_data = inception.preprocess_image(image_data,
                                                IMG_HEIGHT,
                                                IMG_WIDTH,
                                                dt_type == TRAIN)

    # Generate batch
    return _load_batch(image_data, label_data, shuffle=False)
