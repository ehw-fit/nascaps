"""
Learn 
"""

import os
import tensorflow as tf 
import numpy as np
from keras import optimizers

import time
import random
import hw_estimator
from wrapper import CapsNet
import json, gzip
import shutil
from random import randint
import random
from math import ceil
from keras import backend as K
from keras import utils
#from keras.models import Model
#from keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator
import json
import sys
sys.path.append("..")

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU

#config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

fix_out_a=[]
fix_out_b=[]
cross_out=[]

from main import random_configuration, evaluate_population, load_mnist, load_fmnist, load_cifar10, load_cifar100, set_args, set_data, load_svhn

def run_chromosome(chromosome_name, metrics, inshape):
    """ Generates one random configuration and try to evaluate it
    """
    start = time.time()
    random.seed()

    # random initial population Pt
    parent = []
    chromosome = json.load(open(chromosome_name))

    chromosome[-3][-2] = inshape[-1]

    print("parsing chromosome {0}".format(str(chromosome)))
    parent.append({"gene" : chromosome}) # (insize, inchannels, incapsules, n_classes)
    print("\n Evaluate population.\n")

    evaluate_population(parent)

    return parent


if __name__ == "__main__":
    import os
    import argparse
    from keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Random Capsule Network on MNIST.")
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--output', default="results", type=str)
    parser.add_argument('--timeout', default=0, type=int, help="Maximal time in seconds for the training, zero = not set")
    parser.add_argument('--gpus', default=1, type=int)

    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('--cache_dir', default='./cache')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    parser.add_argument('--max_params', default=0, type=int)
                        
    parser.add_argument('chromosome', help="Chromosme")
    args = parser.parse_args()
    set_args(args)
    print(args)


    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)
        
    # load data
    if args.dataset=='mnist':
        (x_train, y_train), (x_test, y_test) = load_mnist()
        inshape=[28, 1, 1, 10]
    elif args.dataset=='fmnist' or args.dataset=='fashion_mnist':
        (x_train, y_train), (x_test, y_test) = load_fmnist()
        inshape=[28, 1, 1, 10]
    elif args.dataset=='cifar10':
        from keras.datasets import cifar10
        (x_train, y_train), (x_test, y_test) = load_cifar10()
        inshape=[32,3,1,10]
    elif args.dataset=='cifar100':
        from keras.datasets import cifar100
        (x_train, y_train), (x_test, y_test) = load_cifar100()
        inshape=[32,3,1,100]
    elif args.dataset=='svhn':
        (x_train, y_train), (x_test, y_test) = load_svhn()
        inshape=[32,3,1,10]
    else:
        (x_train, y_train), (x_test, y_test) = load_mnist()
        inshape=[28, 1, 1, 10]



    set_data(x_train, y_train, x_test, y_test)


    rets = run_chromosome(arg.chromosome, metrics=["accuracy_drop", "energy", "memory", "latency"], inshape=inshape)
    outfile = f"{args.output}_random.json"
    json.dump(rets, open(outfile, "wt"), )
    #tf.app.run()

# define the margin loss like hinge loss
#def margin_loss(y_true, y_pred):
#    lamb, margin = 0.5, 0.1
#    return K.sum(y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
#        1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)
