import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
import shutil

import cocoamodel.configvalues as cv
from cocoamodel import gendataset, configvalues, utils, process
from inception.inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope

slim = tf.contrib.slim

    
def get_vars_to_train_per_depth(depth):
    model_name = 'InceptionResnetV2/'
    # depth = 0 means train all layers.
    # depth = 1 means train only Logits layers.
    # depth = 2 means train Logits and Conv2d_7b_1x1. And so on.
    names = ['Logits', 'Conv2d_7b_1x1', 'Block8', 'Repeat_2', 'Mixed_7a',
             'Repeat_1', 'Mixed_6a', 'Repeat', 'Mixed_5b', 'Conv2d_4a_3x3',
             'Conv2d_3b_1x1', 'Conv2d_2b_3x3', 'Conv2d_2a_3x3', 'Conv2d_1a_3x3']
    selected = names[0:depth]
    vars_to_train =[]
    for i in tf.trainable_variables():
        for layer in selected:
            if i.name.startswith(model_name + layer):
                vars_to_train.append(i)
    return vars_to_train

def trainmode():
    dir_name = utils.gen_dir_name(cv.TRAIN_DIR)
    print(dir_name)
    # Saves the configvalues.py file so we can keep track of all the
    # configuration values that were used.
    # TODO properly export all these configurations.
    shutil.copy('./cocoamodel/configvalues.py', dir_name)

    with tf.Graph().as_default() as graph:
        
        # This can be done in the cpu to save the gpu some memory.
        # Saves about 0,01 second per normal step in a gtx850M2GB
        with tf.device('/cpu:0'):
            images, labels = gendataset.get_batch_input()
    
        process.calc_steps_details(gendataset.count_samples(cv.TRAIN))

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
        
        learn_rate = tf.train.exponential_decay(
            learning_rate = cv.INITIAL_LEARN_RATE,
            global_step = global_step,
            decay_steps = cv.DECAY_STEPS,
            decay_rate = cv.LEARN_DECAY_FACTOR,
            staircase = True)

        optimizer = tf.train.AdamOptimizer(learning_rate = learn_rate)
        
        # Since we do not want to touch the lower layers' weights of this
        # inception model, here we select which variables to train.
        vars_to_train = get_vars_to_train_per_depth(cv.TRAIN_DEPTH)

        train_op = slim.learning.create_train_op(total_loss,
                                                 optimizer,
                                                 variables_to_train=vars_to_train)

        # Set up the metrics.
        tf.summary.scalar('losses/Total_Loss', total_loss)
        tf.summary.scalar('learning_rate', learn_rate)
        metrics_obj = process.Metrics(end_points['Predictions'], labels)

        saver = tf.train.Saver(vars_restore)
        def restore_fn(sess):
            return saver.restore(sess, cv.CHECKPOINT_FILE)
        supervisor = tf.train.Supervisor(logdir = dir_name,
                                         summary_op = None,
                                         init_fn = restore_fn)

        with supervisor.managed_session() as sess:
            # We do the last step out of this for loop.
            metrics_op, summary_op = metrics_obj.prepare_run()
            for step in range(cv.TOTAL_STEPS - 1):
                total_loss, global_step_count, _ = sess.run([train_op,
                                                             global_step,
                                                             metrics_op])
                if step % cv.LOG_FREQ == 0:
                    summaries = sess.run(summary_op)
                    supervisor.summary_computed(sess, summaries)
                    metrics_obj.log_step_loss(total_loss, global_step_count)

                if step % cv.BATCHES_PER_EPOCH == 0:
                    metrics_obj.log_step_loss(total_loss, global_step_count)
                    metrics_obj.log_metrics(sess)
                    supervisor.saver.save(sess,
                                          supervisor.save_path,
                                          global_step = supervisor.global_step)

            total_loss, global_step_count, _ = sess.run([train_op,
                                                         global_step,
                                                         metrics_op])
            metrics_obj.log_step_loss(total_loss, global_step_count)
            metrics_obj.log_metrics(sess)

            supervisor.saver.save(sess,
                                  supervisor.save_path,
                                  global_step = supervisor.global_step)
