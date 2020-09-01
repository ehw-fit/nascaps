# this is a draft of NSGA2 algorith
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf 
import numpy as np
from keras import optimizers

from PIL import Image
import time
import random
import hw_estimator
import svhn
from wrapper import CapsNet
import json, gzip
import shutil
from random import randint
import random
from math import ceil
from keras import backend as K
from keras import utils, callbacks
#from keras.models import Model
#from keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.layer_utils import count_params
from copy import deepcopy
from paretoarchive import PyBspTreeArchive
import uuid
from timeout_callback import TimeoutCallback
import sys
sys.path.append("..")

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


def set_args(a):
    """ Function for setting of global parameters from another modules such as randlearn """
    global args
    args = a


def set_data(x_tr, y_tr, x_te, y_te):
    """ Function for setting of global parameters from another modules such as randlearn """
    global x_train, y_train, x_test, y_test
    global x_train_shapes, x_test_shapes
    x_train, y_train, x_test, y_test = x_tr, y_tr, x_te, y_te
    x_train_shapes = {}
    x_test_shapes = {}

def resize(data_set, size): 
    X_temp = []
    import scipy
    # one channel images: scipy.misc.imresize needs to have an input shape (N, H, W, C) if C is > 1; otherwise only (N, H, W)
    print("input shape", data_set.shape)
    one_channel = False
    if data_set.shape[-1] == 1:
        one_channel = True
        data_set.resize(data_set.shape[:-1])

    print("input shape after reshape", data_set.shape)

    for i in range(data_set.shape[0]):
        resized = scipy.misc.imresize(data_set[i], (size, size))
        X_temp.append(resized)

    X_temp = np.array(X_temp, dtype=np.float32) / 255.

    # if one channel input was resized, add one dimension to the tensor to (N, H, W, 1)
    print("output shape before reshape", X_temp.shape)
    if one_channel:
        X_temp.resize((X_temp.shape[0], size, size, 1))
        data_set.resize(list(data_set.shape) + [1]) # resize dazaset back
    print("output shape after reshape", X_temp.shape)

    return X_temp

fix_out_a=[]
fix_out_b=[]
cross_out=[]

def random_configuration(insize, inchannels, incapsules, n_classes):
    
    N_conv=randint(3, 6)
    N_caps=randint(2, 6)
    N_dcaps=randint(4, 8)
    kernel_sizes=[3, 5, 9]
    withstride2=0
    layer=0
    gene=[]

    resize_input=randint(1,2)
    #resize_input=2
    type=randint(1,2)
    #type=1

    # Convolutional layers
    for i in range(1,N_conv+1):
     
        if i==1:
            insize = insize*resize_input
            inchannels = inchannels
            incapsules = incapsules
            kernsize = 9
            stride = 1
            outsize = ceil(float(insize)/float(stride))
            outchannels = 2**randint(2,6)
            outcapsules = 1             
        else:
            insize = outsize
            inchannels = outchannels
            incapsules = outcapsules
            kernsize = kernel_sizes[randint(0,2)] 
            stride = randint(1,2)
            if stride==2:
                withstride2+=1
            outsize = ceil(float(insize)/float(stride))
            outchannels = 2**randint(2,6)
            outcapsules = 1 
            
        conv = [0, insize, inchannels, incapsules, kernsize, stride, outsize, outchannels, outcapsules]
        #conv = [0, kernsize, stride, outsize, outchannels, outcapsules]
        gene.append(conv)
        layer=layer+1            

        

    # Sabour Capsule layers
    if type==1:
        for i in range(1,N_caps+1):
         
            if i!=N_caps:
                insize = outsize
                inchannels = outchannels
                incapsules = outcapsules
                kernsize = kernel_sizes[randint(0,2)]
                stride = randint(1,2)
                outsize = ceil(float(insize)/float(stride))
                outchannels = 2**randint(2,6)
                outcapsules = min(incapsules + randint(4,20), 64)         
            else:
                insize = outsize
                inchannels = outchannels
                incapsules = outcapsules
                kernsize = insize #kernel_sizes[randint(0,2)]
                stride = randint(1,2)
                outsize = ceil(float(insize)/float(stride))
                outchannels = n_classes
                outcapsules = 2**randint(2,6)
            
            caps = [1, insize, inchannels, incapsules, kernsize, stride, outsize, outchannels, outcapsules]
            #caps = [1, kernsize, stride, outsize, outchannels, outcapsules]
            gene.append(caps)
            layer=layer+1                  
            
    # DeepCaps cell layers
    elif type==2:
        
        w=N_dcaps-4-withstride2
        remaining=N_dcaps
        withstride1 = {}
        for i in range(1,N_dcaps+1):
            if (w>0 and randint(0,2)==1) or remaining<4:
                withstride1[i]=1
                w=w-1
            else:
                withstride1[i]=0
            remaining=remaining-1
            
        for i in range(1,N_dcaps+1):

            if i==1:
                insize = outsize
                inchannels = outchannels
                incapsules = outcapsules
                kernsize = 3                
                if withstride1[i]==1:
                    stride = 1
                else:
                    stride = 2                    
                outsize = (insize + stride - 1) // stride
                outchannels = 32
                outcapsules = 4
                              
            elif i in range(2,N_dcaps):
                insize = outsize
                inchannels = outchannels
                incapsules = outcapsules
                kernsize = 3          
                if withstride1[i]==1:
                    stride = 1
                else:
                    stride = 2                   
                outsize = (insize + stride - 1) // stride
                if outsize==1:
                    stride = 1
                    outsize = (insize + stride - 1) // stride    
                outchannels = 32 # randint(20,50)
                outcapsules = 8

            else:
                insize = outsize
                inchannels = outchannels
                incapsules = outcapsules
                kernsize = insize          
                stride = 1 # unused                 
                outsize = 1
                outchannels = n_classes
                outcapsules = 16
                
            
            d_caps = [2, insize, inchannels, incapsules, kernsize, stride, outsize, outchannels, outcapsules] #Standard CapsCells are used
            #d_caps = [2, kernsize, stride, outsize, outchannels, outcapsules] #Standard CapsCells are used

            gene.append(d_caps)
            layer=layer+1 
 
    N_layers=layer

    if type==2:
        index=layer-1
        print("lengene is "+str(len(gene))+ " layer is "+str(layer))
        while gene[index][1]==gene[index-1][6] and gene[index][1]==gene[index][6] and gene[index][0]==gene[index-1][0]:                  # while gene[index][2]==1         while gene[layer][1]==gene[layer-1][6] and gene[layer][1]==gene[layer][6] and gene[layer][0]==gene[layer-1][0]:
            index-=1
        try:
            xtrachoice=range(index+2,len(gene)-1)  
        except:
            xtrachoice=[-1]

        print("lengene is "+str(len(gene))+ " layer is "+str(layer)+ " index is "+str(index))
        print(xtrachoice)

        try:
            xtraskip = random.choice(xtrachoice) #xtrachoice[randint(0,len(gene)-2-layer)] 
        except:
            xtraskip=-1

        if xtraskip==N_layers:
            xtraskip=-1

    elif type==1 or type==0:
        xtraskip=-1

    gene.append([xtraskip]) 
    gene.append([resize_input]) 

           
    print(gene)
    print("\n")
    return gene

def fix(gene): 

    # Scan for type of layers
    conv=[]
    caps=[]
    dcaps=[]
    N_conv=0
    N_caps=0
    i=0
    type=gene[len(gene)-3][0]

    for layer in gene:
        if layer[0]==0 and len(layer)>1:
            N_conv = N_conv+1
            i+=1
        elif layer[0]==1 and len(layer)>1:
            N_caps = N_caps+1
            i+=1
        elif layer[0]==2 and len(layer)>1:
            N_caps = N_caps+1
            i+=1
    nlayers= N_conv+N_caps

    print("Fixing...")
    count=0
    for layer in range(1,nlayers):

        if len(gene[layer])>1 and gene[layer][0]==0: #convolutional
            #[0, insize, inchannels, incapsules, kernsize, stride, outsize, outchannels, outcapsules]
            gene[layer][1]= gene[layer-1][6]                                       #insize TO DELETE
            gene[layer][2]= gene[layer-1][7]                                       #inchannels TO DELETE
            gene[layer][3]= gene[layer-1][8]                                       #incapsules TO DELETE
            gene[layer][6]= ceil(float(gene[layer][1])/float(gene[layer][5]))      #outsize

        elif len(gene[layer])>1 and gene[layer][0]==1: #sabour caps
            #[1, insize, inchannels, incapsules, kernsize, stride, outsize, outchannels, outcapsules]
            gene[layer][1]= gene[layer-1][6]                                       #insize TO DELETE
            gene[layer][2]= gene[layer-1][7]                                       #inchannels TO DELETE
            gene[layer][3]= gene[layer-1][8]                                       #incapsules TO DELETE
            gene[layer][6]= ceil(float(gene[layer][1])/float(gene[layer][5]))      #outsize

        elif len(gene[layer])>1 and gene[layer][0]==2: #deepcaps
            count+=1
            #[2, insize, inchannels, incapsules, kernsize, stride, outsize, outchannels, outcapsules]
            gene[layer][1]= gene[layer-1][6]                                       #insize TO DELETE
            gene[layer][2]= gene[layer-1][7]                                       #inchannels TO DELETE
            gene[layer][3]= gene[layer-1][8]                                       #incapsules TO DELETE
            gene[layer][6]= (gene[layer][1] + gene[layer][5]- 1) // gene[layer][5] #outsize
            #gene[layer][7]= gene[layer-1][7]
            if count==1:
                gene[layer][8]= 4
            else:
                gene[layer][8]= 8

        else:
            break  

    # Adjust last layer kernel dimension to fit with insize
    gene[len(gene)-3][4] = gene[len(gene)-3][1]

    if type==2:
        layer=len(gene)-3 # posizionato sull'ultimo layer
        print("layer is"+str(layer)+"\n len_gene is"+str(len(gene)))
        print("gene[layer][1]="+str(gene[layer][1]))
        while gene[layer][1]==gene[layer-1][6] and gene[layer][1]==gene[layer][6] and gene[layer][0]==gene[layer-1][0]:
            layer-=1
        print("layer is"+str(layer)+"\n len_gene is"+str(len(gene)))
        try:
            xtrachoice=range(index+2,len(gene)-3)  
        except:
            xtrachoice=[-1]

        print(xtrachoice)

        try:
            xtraskip = random.choice(xtrachoice)
        except:
            xtraskip=-1

        if xtraskip==nlayers-1:
            xtraskip=-1

    elif type==1 or type==0:
        xtraskip=-1

    gene[len(gene)-2][0]=xtraskip

    # resize value adjustes with the one of the first layer
    if gene[0][1]==28 or gene[0][1]==32:
        gene[-1][0]=1
    elif gene[0][1]==56 or gene[0][1]==64:
        gene[-1][0]=2

    print("Fixed.")
    return gene

def crossover(par_a, par_b, cross_out):
    # TODO Add check for same-type of parent networks (in WRAP - to be tested)
    
    numlayers1 = len(par_a)-2        #-1 for xtraskip managment
    numlayers2 = len(par_b)-2    
    cut1=randint(1,numlayers1-1)       # starts from 2 in order to have at least 2 conv layers
    cut2=randint(1,numlayers2-1)
    type=par_a[len(par_a)-3][0]
    print(par_a)
    print(par_b)
    print("crossover")
    countd=0

    if type==1:
        # In order to avoid sequence conv+caps+CONV+CAPS
        while (par_a[cut1-1][0]==1 and par_b[cut2][0]==0) or (par_b[cut1-1][0]==1 and par_a[cut2][0]==0): 
        #while (par_a[cut1-1][0]==1 and par_b[cut2][0]==0) or (par_b[cut1-1][0]==1 and par_a[cut2][0]==0): 
            #cut2=cut2+1  
            cut2=randint(1,numlayers2-1)  

    elif type==2:
        cut1=randint(1,numlayers1-4)
        cut2=randint(1,numlayers2-4)
        # In order to avoid sequence conv+dcaps+CONV+DCAPS
        #while par_a[cut1-1][0]==2 and par_b[cut2][0]==0:         
        while par_a[cut1][0]==2 and par_b[cut2][0]==0:         
            cut2=randint(1,numlayers2-4)
            #cut2=cut2+1    
        #while par_b[cut1-1][0]==2 and par_a[cut2][0]==0:
        while par_b[cut2][0]==2 and par_a[cut1][0]==0:
            cut1=randint(1,numlayers1-4)
            #cut1=cut1+1  

    child_a=deepcopy(par_a[0:cut1])+deepcopy(par_b[cut2:len(par_b)])
    print("\nchild_a: ",child_a)
    child_a=fix(child_a)
    print("\nfchild_a: ",child_a)
    fix_out_a.append(child_a) 
    print("\nfix_out_a: ",fix_out_a)

    child_b=deepcopy(par_b[0:cut2])+deepcopy(par_a[cut1:len(par_a)])
    print("\nchild_b: ",child_b)
    print("\nfix_out_a: ",fix_out_a)

    child_b=fix(child_b)
    print("\nfchild_b: ",child_b)
    print("\nfix_out_a: ",fix_out_a) # At this point the fix_out_a gene has changed
                                     # with the last part coming from fix_out_b
    fix_out_b.append(child_b)
    print("\nfix_out_a: ",fix_out_a) 
    print("\nfix_out_b: ",fix_out_b)

def mutate(gene): # always mutes also xtraskip if type==2
    
    # Scan for type of layers
    N_layers = int(len(gene)-3) 
    conv=[]
    caps=[]
    N_conv=0
    N_caps=0
    i=0
    for layer in gene:
        if layer[0]==0:
            N_conv = N_conv+1
            conv.append(i)
            i+=1
        elif layer[0]==1 or layer[0]==2:
            N_caps = N_caps+1
            caps.append(i)
            i+=1
    
    type=gene[len(gene)-3][0]
    mute_type_choice=[0, 1] # 0 = normal, 1 = delete
    mute_type=random.choice(mute_type_choice)

    if type==2 and N_caps<=5:
        try:
            layer_to_delete_choice=range(1,N_conv)
            layer_to_delete=random.choice(layer_to_delete_choice)
        except:
            mute_type=0 #do not delete
    else:
        try:
            layer_to_delete_choice=range(1,len(gene)-3)
            layer_to_delete=random.choice(layer_to_delete_choice)
        except:
            mute_type=0 #do not delete

    print("mute_type = "+str(mute_type))

    if mute_type==0:
        kernel_sizes=[3, 5, 9]

        choices = [4, 5, 8] #kernelsize, stride, outsize
        layer_to_mutate = randint(0, N_layers)
        el_to_mutate = choices[randint(0,2)]
        current = gene[layer_to_mutate][el_to_mutate]
        new=current
    
        if el_to_mutate==4: #kernel size          if el_to_mutate==1: #kernel size
            while new==current:
                new=kernel_sizes[randint(0,2)] 
        elif el_to_mutate==5: # stride            elif el_to_mutate==2: #stride
            while new==current:
                new=randint(1,2)
        elif el_to_mutate==8: # out_channels      elif el_to_mutate==4: # out_channels
            while new==current:
                new=2**randint(2,6)         

        gene[layer_to_mutate][el_to_mutate]=new
        if el_to_mutate==5:               #       if el_to_mutate==2:
            gene[layer_to_mutate][6] = ceil(float(gene[layer_to_mutate][1])/float(new)) # adjust outsize            ceil(float(gene[layer_to_mutate-1][3])/float(new))
    
        for index in range(layer_to_mutate+1,N_layers-1):
            gene[index][1] = gene[index-1][1] #insize TO DELETE
            gene[index][2] = gene[index-1][2] #inchannel TO DELETE

        if gene[len(gene)-3][0]==2:
            prev_xtraskip=gene[len(gene)-2][0]
            layer=len(gene)-3 # posizionato sull'ultimo layer
            print("layer is"+str(layer)+"\n len_gene is"+str(len(gene)))
            print("gene[layer][1]="+str(gene[layer][1]))
            while gene[layer][1]==gene[layer-1][6] and gene[layer][1]==gene[layer][6] and gene[layer][0]==gene[layer-1][0]:
                layer-=1
            print("layer is"+str(layer)+"\n len_gene is"+str(len(gene)))
            try:
                xtrachoice=range(index+2,len(gene)-3)  
            except:
                xtrachoice=[-1]

            print(xtrachoice)

            try:
                xtraskip = random.choice(xtrachoice) 
            except:
                xtraskip=-1

            if xtraskip==N_layers:
                xtraskip=-1

        elif gene[len(gene)-3][0]==1 or gene[len(gene)-3][0]==0:
            xtraskip=-1

        gene[len(gene)-2][0]=xtraskip
   
    elif mute_type==1:
        del gene[layer_to_delete]
        
    gene=deepcopy(fix(gene))
        
    
    return gene



def evaluate_population(pop):
    """ 
        Most important part; needs to be implemented! 
        population is a list of individuals where each individual is a dictionary
        each individual contains at least "gene" item; the rest of them is calculated lated
    """
    popcnt = len(pop)
    for popid, p in enumerate(pop):
        print(f"### EVAL candidate {popid}/{popcnt}")
        cache_name = "cache_"+str(args.dataset)+"_e"+str(args.epochs) 
        try:
            cache = json.load(gzip.open(args.cache_dir + "/%s.json.gz" % cache_name, "rt"))
        except IOError:
            cache = {} 

        assert "gene" in p
        #p["gene"]=[[0, 64, 3, 1, 3, 1, 64, 128, 1], [2, 64, 32, 4, 3, 2, 32, 32, 4], [2, 32, 32, 4, 3, 2, 16, 32, 8], [2, 16, 32, 8, 3, 2, 8, 32, 8], [2, 8, 32, 8, 3, 2, 4, 32, 8], [2, 4, 32, 8, 4, 1, 1, 10, 16], [4], [2]]
        if ("accuracy_drop" not in p) or ("gene" not in cache):
            p["runid"], train_acc = wrap_train_test(p["gene"])
            p["accuracy_drop"] = 1 - train_acc
            from_cache = False
            # note: we are minimizing all parameters! therefore accuracy drop must be evaluated instead of the accuracy
        else:
             train_acc=cache[genestr(p["gene"])] 
             from_cache = True

        cache[genestr(p["gene"])] = train_acc
        print("\nTrain accuracy: "+ str(train_acc)+"\n\n")
        
        print("\nHW Estimator\n")
        estimator = hw_estimator.hw_estimator()
        
        estimator.parse_genotype(p["gene"])
        
        print("\nParsed gene\n")
        
        if "energy" not in p:
            p["energy"] = estimator.get_energy()

        if "latency" not in p:
            p["latency"] = estimator.get_latency()
            
        if "memory" not in p:
            p["memory"] = estimator.get_memory()
        
        for x in ["memory", "latency", "accuracy_drop", "energy"]:
            if x in p:
                print("OPT eval", x, "=", p[x])

        # save the previous version of the cache
        # because of concurrent processes (it may destroy the file)
        if not from_cache: # result was not cached
            backupid = 0
            while(os.path.isfile(args.cache_dir + "/backup_%s_%03d.json.gz" % (cache_name, backupid))):
                backupid += 1
            try:
                shutil.copy(args.cache_dir + "/cache_"+str(args.dataset)+"_e"+str(args.epochs)+".json.gz", args.cache_dir + "/backup_%s_%03d.json.gz" % (cache_name, backupid))
            except: 
                pass 
            
            # save the cache
            json.dump(cache, gzip.open(args.cache_dir + "/%s.json.gz" % cache_name, "wt" ), indent=2)


    print("\nEvaluation of current population completed\n")
    
    return pop 


def genestr(gene):
    return str(gene).replace(" ", "")

def wrap_train_test(gene):
    global x_train, y_train, x_test, y_test
    global x_train_shapes, x_test_shapes
    runid = "N/A"
    print(gene)

    with open("tested.log", "a") as f:
        f.write(genestr(gene))
        f.write("\n")
    
    print("\nWrapping...\n")
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        

    # reshaping of the training data
    if gene[-1][0]==2: # reshaping is enabled
        desired_size = gene[0][1]
        if desired_size not in x_train_shapes:
            x_train_shapes[desired_size] = resize(x_train, desired_size)
        if desired_size not in x_test_shapes:
            x_test_shapes[desired_size] = resize(x_test, desired_size)
        
        x_train_current = x_train_shapes[desired_size]
        x_test_current = x_test_shapes[desired_size]
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


    trainable_count = count_params(model.trainable_weights)
    if args.max_params > 0 and trainable_count > args.max_params:
        print(f"## ERR: number of trainable params {trainable_count} exceeded limit {args.max_params}")
        tf.keras.backend.clear_session()
        K.clear_session()
        return runid, 0

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
    return runid, test_acc



    
def crowding_distance(par, objs):
    """ calculates crowding distance for pareto frontier par for objectives objs """
    park = list(zip(range(len(par)), par)) # list of "ids, evaluated_offsprint"
    distance = [0 for _ in range(len(par))]

    for o in objs:
        sval = sorted(park, key=lambda x: x[1][o]) # sort by objective
        minval, maxval = sval[0][1][o], sval[-1][1][o]
        # distance of the lowest and highest value is infty
        distance[sval[0][0]] = float("inf")
        distance[sval[-1][0]] = float("inf") 

        for i in range(1, len(sval) - 1):
            distance[sval[i][0]] += abs(sval[i - 1][1][o] - sval[i + 1][1][o]) / (maxval - minval)
   
    return zip(par, distance)

def crowding_reduce(par, number, objs):
    """ Removes the elements from pareto frontier; iteratively removest the node with the lowest crowding distance """
    par = par
    while len(par) > number:
        vals = crowding_distance(par, objs)
        vals = sorted(vals, key = lambda x: -x[1]) # sort by distance descending
        #print(vals)

        par = [x[0] for x in vals[:-1]]
    return par

def run_NSGA2(metrics, inshape, p_size = 2, q_size = 2, generations=5, mutation_rate = 0.1):
    """ Heuristic optimization NSGA2 algorithm; 
        metrics - which parameters should be optimized, see evaluate_population
        p_size - number of elements in the parent generation
        q_size - number of offsprings generated by binary crossover and random mutation
        qenerations - number of generations
        mutation_rate - probability of mutation of one configuration parameter
    """
    assert len(metrics) >= 2
    start = time.time()
    random.seed()

    # random initial population Pt
    parent = []
    for i in range(p_size):
        parent.append({"gene" : random_configuration(inshape[0], inshape[1], inshape[2], inshape[3]) }) # (insize, inchannels, incapsules, n_classes)
    print("\n Evaluate population.\n")

    evaluate_population(parent)

    # genetic loop
    print("\nGenetic Loop.\n")
    for g in range(generations):
        # generate offsprings Qt
        offsprings = []
        for i in range(int(q_size/2)):
            par_a = random.choice(parent)
            print(par_a)
            print(par_a["gene"][len(par_a["gene"])-3][0])
            par_b = random.choice(parent)
            print(par_b["gene"][len(par_b["gene"])-3][0])

            # todo: this part can goes to infinite loop
            cnt=0
            while par_b["gene"][:]==par_a["gene"][:] or par_a["gene"][len(par_a["gene"])-3][0]!=par_b["gene"][len(par_b["gene"])-3][0] or cnt<100:
                print(par_a["gene"][len(par_a["gene"])-3][0])
                par_b = random.choice(parent)
                cnt+=1
                if cnt==99:
                    break

            if cnt>98:
                mut=[[],[]]
                # no second parent found in 100 tries
                mut[0]=mutate(par_a["gene"])
                mut[1]=mutate(par_a["gene"])
                for off in mut:               
                    c = {}
                    c["gene"] = off
                    offsprings.append(c)

            else: 
                print(par_b)
                # binary crossover
                crossover(par_a["gene"], par_b["gene"], cross_out)
                out_cross=[fix_out_a[0], fix_out_b[0]]
                print("out_cross: "+ str(out_cross))
                #print("fa: "+ str(child_a))
                print("out of crossover\n")
                for off in out_cross:               
                    c = {}
                    c["gene"] = off

                    # mutation
                    if random.random() < mutation_rate:
                        mutate(c["gene"]) 
                
                    offsprings.append(c)

        evaluate_population(offsprings)
        del fix_out_a[:]
        del fix_out_b[:]
        del cross_out[:]


        population = parent + offsprings

        # selection of pareto frontiers
        next_parent = []
        while len(next_parent) < p_size:
            # select pareto frontier
            pareto = PyBspTreeArchive(len(metrics)).filter([[x[m] for m in metrics] for x in population], returnIds=True)

            current_pareto = [population[i] for i in pareto]
            missing = p_size - len(next_parent)

            if(len(current_pareto) <= missing): # can we put all pareto frontier to the next parent
                next_parent += current_pareto
            else: # distance crowding 
                next_parent += crowding_reduce(current_pareto, missing, metrics)

            for i in reversed(sorted(pareto)): # delete nodes from the current population
                population.pop(i)

        parent = next_parent
        print("gen ", g)

        # TODO: I recomend to save the current population to some TMP folder; the name of the results must be unique for each separate run
        # because many runs will run in parallel
        json.dump(parent, open(f"{args.save_dir}/{args.output}_gen_{g}.json", "wt"))
    # final filtering
    evaluate_population(parent)
    pareto = PyBspTreeArchive(len(metrics)).filter([[x[m] for m in metrics] for x in parent], returnIds=True)

    ret = []
    for i in pareto:
        ret.append(parent[i])

    print("Filt pareto combinations: %d in %f seconds" % (len(ret), time.time() - start))

    return ret

#def main(argv=None):
    # this is the very main script!
#    rets = run_NSGA2(metrics=["accuracy_drop", "energy", "memory", "latency"])
#    # todo : save return population as json
#    outfile = "results.json"
#    json.dump(rets, open(outfile, "wt"), )

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
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    timeout_call = TimeoutCallback(args.timeout) # timeout

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
    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
                        steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                        epochs=args.epochs,
                        validation_data=[[x_test, y_test], [y_test, x_test]],
                        callbacks=[timeout_call, log, checkpoint, lr_decay])
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




def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = utils.to_categorical(y_train.astype('float32'))
    y_test = utils.to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)


def load_fmnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = utils.to_categorical(y_train.astype('float32'))
    y_test = utils.to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)

def load_cifar10():
    from keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()  
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    y_train = utils.to_categorical(y_train.astype('float32'))
    y_test = utils.to_categorical(y_test.astype('float32'))
    
    # data preprocessing 
    x_train[:,:,:,0] = (x_train[:,:,:,0]-123.680)
    x_train[:,:,:,1] = (x_train[:,:,:,1]-116.779)
    x_train[:,:,:,2] = (x_train[:,:,:,2]-103.939)
    x_test[:,:,:,0] = (x_test[:,:,:,0]-123.680)
    x_test[:,:,:,1] = (x_test[:,:,:,1]-116.779)
    x_test[:,:,:,2] = (x_test[:,:,:,2]-103.939)
    
    return (x_train, y_train), (x_test, y_test)



def load_cifar100():
    from keras.datasets import cifar100
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    y_train = utils.to_categorical(y_train.astype('float32'))
    y_test = utils.to_categorical(y_test.astype('float32'))
        
    # data preprocessing 
    x_train[:,:,:,0] = (x_train[:,:,:,0]-123.680)
    x_train[:,:,:,1] = (x_train[:,:,:,1]-116.779)
    x_train[:,:,:,2] = (x_train[:,:,:,2]-103.939)
    x_test[:,:,:,0] = (x_test[:,:,:,0]-123.680)
    x_test[:,:,:,1] = (x_test[:,:,:,1]-116.779)
    x_test[:,:,:,2] = (x_test[:,:,:,2]-103.939)
    
    return (x_train, y_train), (x_test, y_test)


def load_svhn():
    (x_train, y_train), (x_test, y_test) = svhn.load_data()  
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
        
    return (x_train, y_train), (x_test, y_test)




if __name__ == "__main__":
    import os
    import argparse

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--population', default=2, type=int)
    parser.add_argument('--offsprings', default=2, type=int)
    parser.add_argument('--generations', default=5, type=int)
    parser.add_argument('--output', default="results", type=str)
    parser.add_argument('--timeout', default=0, type=int, help="Maximal time in seconds for the training, zero = not set")
    parser.add_argument('--gpus', default=1, type=int)

    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--max_params', default=20000000, type=int)
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
    parser.add_argument('--debug', action='store_true', help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('--cache_dir', default='./cache')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    #parser.add_argument('--digit', default=5, type=int, help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
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

    # cache for reshaped inputs
    x_train_shapes = {}
    x_test_shapes = {}



    rets = run_NSGA2(metrics=["accuracy_drop", "energy", "memory", "latency"], inshape=inshape, p_size=args.population, q_size=args.offsprings, generations=args.generations)
    outfile = f"{args.output}_results.json"
    json.dump(rets, open(outfile, "wt"), )



