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
    #filename = '/home/july/CocoaModel/dataset_raw/images/image_jpg/img9.jpg'
    x = evaluate.test_image(filename)
    if x[0] > x[1]:
        print('\n\n\nI am %.2f percent sure that this is a cocoa pod.\n\n\n' % (float(x[0])*100))
    else:
        print('\n\n\nI don\'t think there is cocoa in this photo. %.2f \n\n\n' % (float(x[1])*100))
    #print(repr(x))

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--trainmode', action='store_true',
                    help='Train mode on.')
    parser.add_argument('--evalmode', type=str,
                    help='Eval mode on.')
    parser.add_argument('--is_cocoa', type=str,
                    help='Is there a cocoa in this image?')

    args = parser.parse_args()

    utils.config_logging()

    if args.trainmode:
        training.trainmode()
        exit()

    if args.evalmode:
        evaluate.evalmode(args.evalmode)
        exit()

    if args.is_cocoa:
        cocoa_or_not(args.is_cocoa)
        exit()


if __name__ == '__main__':
    main()
