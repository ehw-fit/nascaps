"""
Complex training for CIFAR10
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

from main import load_mnist, load_fmnist, load_cifar10, load_cifar100, set_args, set_data, load_svhn
from main import resize


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

def train(model, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks

    import uuid
    runid = uuid.uuid1().hex
    print("### runid:", runid)
    
    log = callbacks.CSVLogger(args.save_dir + '/' + runid + '.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))

    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/' + runid + '.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    steps_per_epoch=int(y_train.shape[0] / args.batch_size)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** (epoch * steps_per_epoch / args.decay_steps)))


    try:
        gpus = args.gpus
        if not gpus:
            gpus = tf.contrib.eager.num_gpus()

        if gpus > 1:
            model = utils.multi_gpu_model(model, gpus, cpu_merge=False)
            print(f"Training using multiple GPUs.. ({gpus})")
        else:
            print("Single-GPU model is used")
    except Exception as e:
        print("Exception ", e)
        print("Training using single GPU or CPU..")

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})

    """
    # Training without data augmentation:
    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
    """

    # Begin: Training with data augmentation ---------------------------------------------------------------------#
    def train_generator(x, y, batch_size, shift_fraction=0., rotation_range=0., horizontal_flip=False):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction, # shift up to 2 pixel for MNIST
                                           rotation_range = rotation_range, 
                                           horizontal_flip = horizontal_flip)  
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, args.shift_fraction, args.rotation_range, args.horizontal_flip),
                        steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                        epochs=args.epochs,
                        validation_data=[[x_test, y_test], [y_test, x_test]],
                        callbacks=[log, checkpoint, lr_decay])
    # End: Training with data augmentation -----------------------------------------------------------------------#

    #model.save_weights(args.save_dir + '/trained_model.h5')
    #print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    #from keras.utils import plot_log
    #plot_log(args.save_dir + '/log.csv', show=True)

    return runid, model



def test(model, data, args):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    print('-'*30 + 'Begin: test' + '-'*30)
    test_acc= np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0]
    print('Test acc:', test_acc)
    return test_acc



def wrap_train_test(gene):
    global x_train, y_train, x_test, y_test
    runid = "N/A"
    print(gene)

    print("\nWrapping...\n")
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        

    # reshaping of the training data
    if gene[-1][0]==2: # reshaping is enabled
        desired_size = gene[0][1]
        x_train_current = resize(x_train, desired_size)
        x_test_current = resize(x_test, desired_size)
    elif gene[-1][0]==1: # no reshaping
        x_train_current = x_train
        x_test_current = x_test
    else:
        print("#### INVALID GENE - last value is not 1 nor 2", gene[-1][0])
        return runid, 0


    # define model
    try:
        print("x_train shape: "+ str(x_train_current.shape[1:]))
        model, eval_model, manipulate_model = CapsNet(gene = gene, input_shape=x_train_current.shape[1:],
                                                  n_class=len(np.unique(np.argmax(y_train, 1))),
                                                  routings=args.routings)
    except ValueError as e: # some bug in the chromosome ....
        print("#### VALUE error desc ", e)
        print("#### VALUE error gene ", gene)
        tf.keras.backend.clear_session()
        K.clear_session()
        return runid, 0
    except tf.errors.ResourceExhaustedError as e: # some bug in the chromosome ....
        print("#### Out of resources error desc ", e)
        print("#### Out of resources error gene ", gene)
        tf.keras.backend.clear_session()
        K.clear_session()
        return runid, 0


    model.summary()

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if not args.testing:
        # if gene[len(gene)-1][0]==2:
        #     x_train = resize(x_train, gene[0][1]) #64
        #     x_test = resize(x_test, gene[0][1])
        #     train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
        # elif gene[len(gene)-1][0]==1:
        print("Train shapes:", x_train.shape, y_train.shape)
        runid, _ = train(model=model, data=((x_train_current, y_train), (x_test_current, y_test)), args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
    test_acc = test(model=eval_model, data=(x_test_current, y_test), args=args)
   
    tf.keras.backend.clear_session()
    K.clear_session()
    if False:
        import os
        os.system("nvidia-smi")
    return runid, test_acc


def run_chromosome(chromosome_name, metrics, inshape):
    """ Train and tests given chromosome
    """
    start = time.time()
    random.seed()

    # random initial population Pt
    parent = []
    chromosome = json.load(open(chromosome_name))

    chromosome[-3][-2] = inshape[-1]

    print("parsing chromosome {0}".format(str(chromosome)))
    accuracy, _ = wrap_train_test(chromosome)

    return accuracy


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
    parser.add_argument('--decay_steps', default=2000, type=float,
                        help="Decay steps for the lr schedule")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--rotation_range', default=0., type=float,
                        help="Degrees of rotation of input images")
    parser.add_argument('--horizontal_flip', default=False, action='store_true',
                        help="Apply random horizontal flip to input images")                    
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


    rets = run_chromosome(args.chromosome, metrics=["accuracy_drop", "energy", "memory", "latency"], inshape=inshape)
    #outfile = f"{args.output}_random.json"
    #json.dump(rets, open(outfile, "wt"), )
    #tf.app.run()

# define the margin loss like hinge loss
#def margin_loss(y_true, y_pred):
#    lamb, margin = 0.5, 0.1
#    return K.sum(y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
#        1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)
