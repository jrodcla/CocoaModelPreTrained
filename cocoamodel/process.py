
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

        self.fneg, fneg_update = \
            tf.contrib.metrics.streaming_false_negatives(self.pred, self.labels)
        self.fpos, fpos_update = \
            tf.contrib.metrics.streaming_false_positives(self.pred, self.labels)
        self.tneg, tneg_update = \
            tf.contrib.metrics.streaming_true_negatives(self.pred, self.labels)
        self.tpos, tpos_update = \
            tf.contrib.metrics.streaming_true_positives(self.pred, self.labels)

        self.mcc = tf.Variable(tf.zeros([]), name='mcc')
        mcc_update = tf.assign(self.mcc,
                               tf.divide(
                                   tf.subtract(
                                       tf.multiply(self.tpos, self.tneg),
                                       tf.multiply(self.fpos, self.fneg)),
                                   tf.sqrt(
                                       tf.multiply(
                                           tf.multiply(
                                               tf.add(self.tpos, self.fpos),
                                               tf.add(self.tpos, self.fneg)),
                                           tf.multiply(
                                               tf.add(self.tneg, self.fpos),
                                               tf.add(self.tneg, self.fneg))))))

        self.metrics_op = tf.group(accuracy_update,
                                   precision_update,
                                   self.prob,
                                   recall_update,
                                   fneg_update,
                                   fpos_update,
                                   tneg_update,
                                   tpos_update,
                                   mcc_update)
        return self.metrics_op

    def __add_to_summary(self):
        tf.summary.histogram('Probabilities', self.prob)
        tf.summary.scalar('Accuracy', self.accuracy)
        tf.summary.scalar('Precision', self.precision)
        tf.summary.scalar('Recall', self.recall)
        tf.summary.scalar('True_Pos', self.tpos)
        tf.summary.scalar('False_Pos', self.fpos)
        tf.summary.scalar('True_Neg', self.tneg)
        tf.summary.scalar('False_Neg', self.fneg)
        tf.summary.scalar('MCC', self.mcc)
        self.summary_op = tf.summary.merge_all()
        return self.summary_op

    def log_metrics(self, sess):
        logging.info('Accuracy  : %.3f', sess.run(self.accuracy))
        logging.info('Precision : %.3f', sess.run(self.precision))
        logging.info('Recall    : %.3f', sess.run(self.recall))
        logging.info('True Pos  : %.3f', sess.run(self.tpos))
        logging.info('False Pos : %.3f', sess.run(self.fpos))
        logging.info('True Neg  : %.3f', sess.run(self.tneg))
        logging.info('False Neg : %.3f', sess.run(self.fneg))
        logging.info('MCC       : %.3f', sess.run(self.mcc))

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
