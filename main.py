#!/usr/bin/python3.5
import argparse
from datetime import datetime
import time
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
import shutil

import cocoamodel.configvalues as cv
from cocoamodel import gendataset, configvalues, utils, process, training, evaluate

slim = tf.contrib.slim

def cocoa_or_not(filename):
    x = evaluate.test_image(filename)
    if cv.CLASSES == 2:
        if x[0] > x[1]:
            print('\n\n\nPresence of Cocoa: %.2f \n\n\n' % (float(x[0])*100))
        else:
            print('\n\n\nAbsence of Cocoa: %.2f \n\n\n' % (float(x[1])*100))
    else:
        for i in range(0, len(x)):
            print ('\n\n\nClass %d: %.2f \n\n\n' % (i, float(x[i])*100))


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--trainmode', action='store_true',
                    help='Train mode on.')
    parser.add_argument('--evalmode', type=str,
                    help='Eval mode on.')
    parser.add_argument('--is_cocoa', type=str,
                    help='Is there a cocoa in this image?')
    parser.add_argument('--check_dir', type=str,
                    help='Directory where to take the checkpoint from.')

    args = parser.parse_args()

    utils.config_logging()

    if args.trainmode:
        training.trainmode()
        exit()

    if args.evalmode:
        evaluate.evalmode(args.evalmode)
        exit()

    if args.is_cocoa:
        if args.check_dir:
            try:
                cv.CHECKPOINT_FILE = tf.train.latest_checkpoint(args.check_dir)
            except:
                logging.info("Checkpoint not found. Check the directory provided.")
                exit()
        cocoa_or_not(args.is_cocoa)
        exit()


if __name__ == '__main__':
    main()
