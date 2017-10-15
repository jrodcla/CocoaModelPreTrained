from datetime import datetime
import os
import tensorflow as tf

from cocoamodel.configvalues import *

def current_time():
    now = datetime.now()
    if now.day in (1,9):
        pattern = '%s-%s-0%s-%s-%s'
    else:
        pattern = '%s-%s-%s-%s-%s'
    return pattern % (now.year, now.month, now.day, now.hour, now.minute)

def gen_dir_name(base_name):
    dir_name = base_name + '-' + current_time()
    try:
        if tf.gfile.Exists(dir_name):
            tf.gfile.DeleteRecursively(dir_name)
        tf.gfile.MakeDirs(dir_name)
    except Exception as e:
        raise e
    else:
        return dir_name

def config_logging():
    ##TODO Check if I'm still using this dir at all
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    tf.logging.set_verbosity(tf.logging.INFO)
