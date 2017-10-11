import argparse
from datetime import datetime
import os
import time
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
import shutil

from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import cocoamodel.configvalues as cv
from cocoamodel import gendataset, configvalues, utils

slim = tf.contrib.slim

def trainmode():
    dir_name = utils.gen_dir_name(cv.TRAIN_DIR)
    # Saves the configvalues.py file so we can keep track of all the
    # configuration values that were used.
    # TODO proper export all these configurations.
    shutil.copy('./cocoamodel/configvalues.py', cv.TRAIN_DIR)

    with tf.Graph().as_default() as graph:
        
        #TODO maybe use with tf.device('/cpu:0'): ?
        images, labels = gendataset.get_batch_input()
    
        cv.EXAMPLES_PER_EPOCH = gendataset.count_samples(cv.TRAIN)
        cv.DECAY_STEPS = int((cv.EXAMPLES_PER_EPOCH / cv.BATCH_SIZE) 
                             * cv.EPOCHS_BEFORE_DECAY)
        cv.TOTAL_STEPS = cv.EXAMPLES_PER_EPOCH * cv.NUM_EPOCHS

        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            logits, end_points = inception_resnet_v2(images,
                                                     num_classes = cv.CLASSES,
                                                     is_training = True)

        #  We are not training the original ImageNet which the checkpoint was
        # trained to, so we should exclude the Logits scopes, since the number
        # of classes are obviously different.
        vars_restore = slim.get_variables_to_restore(
            exclude = ['InceptionResnetV2/Logits',
                       'InceptionResnetV2/AuxLogits'])

        loss = tf.losses.softmax_cross_entropy(
            onehot_labels = slim.one_hot_encoding(labels, cv.CLASSES),
            logits = logits)

        total_loss = tf.losses.get_total_loss()

        global_step = tf.train.get_or_create_global_step()
        
        learn_rate = lr = tf.train.exponential_decay(
            learning_rate = cv.INITIAL_LEARN_RATE,
            global_step = global_step,
            decay_steps = cv.DECAY_STEPS,
            decay_rate = cv.LEARN_DECAY_FACTOR,
            staircase = True)

        optimizer = tf.train.AdamOptimizer(learning_rate = learn_rate)
        
        # Since we do not want to touch the lower layers' weights of this
        # inception model, here we select which variables to train.
        vars_to_train =[]
        for i in tf.trainable_variables():
            if i.name.startswith('InceptionResnetV2/Conv2d_7b_1x1/') \
            or i.name.startswith('InceptionResnetV2/Logits/Logits/'):
                vars_to_train.append(i)

        train_op = slim.learning.create_train_op(total_loss,
                                                 optimizer,
                                                 variables_to_train=vars_to_train)

        # Set up the metrics.
        predictions = tf.argmax(end_points['Predictions'], 1)
        probabilities = end_points['Predictions']
        accuracy, accuracy_update = \
            tf.contrib.metrics.streaming_accuracy(predictions, labels)
        precision, precision_update = \
            tf.contrib.metrics.streaming_precision(predictions, labels)
        metrics_op = tf.group(accuracy_update, precision_update, probabilities)

        tf.summary.scalar('losses/Total_Loss', total_loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('precision', precision)
        tf.summary.scalar('learning_rate', lr)
        my_summary_op = tf.summary.merge_all()

        saver = tf.train.Saver(vars_restore)
        def restore_fn(sess):
            return saver.restore(sess, cv.CHECKPOINT_FILE)
        supervisor = tf.train.Supervisor(logdir = cv.LOG_DIR,
                                         summary_op = None,
                                         init_fn = restore_fn)

        with supervisor.managed_session() as sess:
            # We do the last step out of this for loop.
            start_time = time.time()
            for step in range(cv.TOTAL_STEPS - 1):
                total_loss, global_step_count, _ = sess.run([train_op,
                                                             global_step,
                                                             metrics_op])
                if step % cv.LOG_FREQ == 0:
                    summaries = sess.run(my_summary_op)
                    supervisor.summary_computed(sess, summaries)
                    time_elapsed = time.time() - start_time
                    logging.info('%s - Step %s: Loss: %.4f (%.2f sec/step)',
                         datetime.now(),
                         global_step_count,
                         total_loss,
                         time_elapsed/cv.LOG_FREQ)
                    start_time = time.time()

            last_loss = train_step(sess, train_op, supervisor.global_step)
            logging.info('Final Loss: %s', last_loss)
            logging.info('Final Accuracy: %s', sess.run(accuracy))
            logging.info('Final Precision: %s', sess.run(precision))
            supervisor.saver.save(sess,
                                  supervisor.save_path,
                                  global_step = supervisor.global_step)




def evalmode():
    dir_name = utils.gen_dir_name(cv.EVAL_DIR)
    pass

def cocoa_or_not():
    pass

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--trainmode', action='store_true',
                    help='Train mode on.')
    parser.add_argument('--evalmode', action='store_true',
                    help='Eval mode on.')
    parser.add_argument('--is_cocoa', type=str,
                    help='Is there a cocoa in this image?')

    args = parser.parse_args()

    utils.config_logging()

    if args.trainmode:
        trainmode()
        exit()

    if args.evalmode:
        evalmode()
        exit()

    if args.is_cocoa:
        cocoa_or_not()
        exit()


if __name__ == '__main__':
    main()
