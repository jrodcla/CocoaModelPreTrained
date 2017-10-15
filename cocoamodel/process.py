
from datetime import datetime

from tensorflow.python.platform import tf_logging as logging
import tensorflow as tf
import time

import cocoamodel.configvalues as cv

def calc_steps_details(total_examples):
    cv.EXAMPLES_PER_EPOCH = total_examples
    cv.BATCHES_PER_EPOCH = int(cv.EXAMPLES_PER_EPOCH / cv.BATCH_SIZE)
    cv.DECAY_STEPS = cv.BATCHES_PER_EPOCH * cv.EPOCHS_BEFORE_DECAY
    cv.TOTAL_STEPS = cv.BATCHES_PER_EPOCH * cv.NUM_EPOCHS


class Metrics:
    def __init__(self, probabilities, labels):
        self.prob = probabilities
        self.labels = labels
        self.pred = tf.argmax(self.prob, 1)
        self.__gen_metrics()
        self.__add_to_summary()

    def __gen_metrics(self):
        self.accuracy, accuracy_update = \
            tf.contrib.metrics.streaming_accuracy(self.pred, self.labels)
        self.precision, precision_update = \
            tf.contrib.metrics.streaming_precision(self.pred, self.labels)
        self.recall, recall_update = \
            tf.contrib.metrics.streaming_recall(self.pred, self.labels)
        self.metrics_op = tf.group(accuracy_update,
                                   precision_update,
                                   self.prob,
                                   recall_update)
        return self.metrics_op

    def __add_to_summary(self):
        tf.summary.histogram('Probabilities', self.prob)
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('precision', self.precision)
        tf.summary.scalar('recall', self.recall)
        self.summary_op = tf.summary.merge_all()
        return self.summary_op

    def log_metrics(self, sess):
        logging.info('Accuracy  : %.3f', sess.run(self.accuracy))
        logging.info('Precision : %.3f', sess.run(self.precision))
        logging.info('Recall    : %.3f', sess.run(self.recall))

    def log_step_accuracy(self, sess, step):
        value = sess.run(self.accuracy)
        self.log_step_x('Accuracy', value, step)

    def log_step_loss(self, value, step):
        self.log_step_x('Loss', value, step)
        
    def log_step_x(self, name, value, step):
        time_elapsed = time.time() - self.start_time
        logging.info('%s - %s/%s - %s: %.4f - %.2f sec/step',
                                datetime.now(),
                                step,
                                cv.TOTAL_STEPS,
                                name,
                                value,
                                time_elapsed/cv.LOG_FREQ)
        self.start_time = time.time()
        
        
    def prepare_run(self):
        self.start_time = time.time()
        return self.metrics_op, self.summary_op
