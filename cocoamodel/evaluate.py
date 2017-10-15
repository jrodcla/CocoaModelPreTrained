import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
import shutil

import cocoamodel.configvalues as cv
from cocoamodel import gendataset, configvalues, utils, process
from inception.inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import inception.inception_preprocessing as inception
slim = tf.contrib.slim

def evalmode(chkpt_dir):
    dir_name = utils.gen_dir_name(cv.EVAL_DIR)
    try:
        cv.CHECKPOINT_FILE = tf.train.latest_checkpoint(chkpt_dir)
    except:
        logging.info("Checkpoint not found. Check the directory provided.")
        exit()

    with tf.Graph().as_default() as graph:
        with tf.device('/cpu:0'):
            images, labels = gendataset.get_batch_input(cv.VALID)

        # With no training load, the network runs way faster and consumes less
        # memory, so the batch size can be significantly larger. One epoch is
        # enough, since it will run through all the data.
        cv.BATCH_SIZE = 3 * cv.BATCH_SIZE
        cv.NUM_EPOCHS = 1

        process.calc_steps_details(gendataset.count_samples(cv.VALID))

        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            logits, end_points = inception_resnet_v2(images,
                                                     num_classes = cv.CLASSES,
                                                     is_training = False)

        # Here we restore all the variables.
        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        def restore_fn(sess):
            return saver.restore(sess, cv.CHECKPOINT_FILE)

        global_step = tf.train.get_or_create_global_step()
        global_step_op = tf.assign(global_step, global_step + 1)

        # Set up the metrics.
        metrics_obj = process.Metrics(end_points['Predictions'], labels)

        supervisor = tf.train.Supervisor(logdir = dir_name,
                                         summary_op = None,
                                         saver = None,
                                         init_fn = restore_fn)

        with supervisor.managed_session() as sess:
            # We do the last step out of this for loop.
            metrics_op, summary_op = metrics_obj.prepare_run()
            for step in range(cv.TOTAL_STEPS - 1):
                sess.run(supervisor.global_step)
                global_step_count, _ = sess.run([global_step_op, metrics_op])
                summaries = sess.run(summary_op)
                supervisor.summary_computed(sess, summaries)
                if step % cv.LOG_FREQ == 0:
                    metrics_obj.log_step_accuracy(sess, global_step_count)

            sess.run([global_step_op, metrics_op])

            metrics_obj.log_metrics(sess)
            
            
def convert_image(filename):
    img = tf.read_file(filename)
    image_data = tf.convert_to_tensor(img)
    image_data = tf.image.decode_jpeg(image_data, channels=3)
    image_data = tf.expand_dims(image_data, 0)
    image_data = tf.image.convert_image_dtype(image_data, dtype=tf.float32)
    image_data = tf.image.resize_nearest_neighbor(image_data, 
                                                    [cv.IMG_HEIGHT,
                                                    cv.IMG_WIDTH])
    image_data = tf.squeeze(image_data)
    image_data = tf.expand_dims(image_data, 0)
    return image_data


def test_image(filename):
    image_tensor = convert_image(filename)
    sess = tf.InteractiveSession()
    graph = tf.Graph()
    graph.as_default()

    with slim.arg_scope(inception_resnet_v2_arg_scope()):
        logits, _ = inception_resnet_v2(image_tensor, num_classes=2, is_training=False)
        probabilities = tf.nn.softmax(logits)
        init_fn = slim.assign_from_checkpoint_fn(
                cv.CHECKPOINT_FILE,
                slim.get_model_variables('InceptionResnetV2'))
        init_fn(sess)
        np_image, probabilities = sess.run([image_tensor, probabilities])
        return probabilities[0, 0:]
