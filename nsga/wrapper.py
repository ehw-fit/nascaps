# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 09:47:35 2020

@author: MA
"""
    


from keras import layers, models#, optimizers
from keras import backend as K
#from keras.utils import to_categorical
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from layers.DeepCapsKerasLayers import Conv2DCaps, ConvCapsuleLayer3D, FlattenCaps, DClassCaps, ConvertToCaps
from keras.layers import BatchNormalization
K.set_image_data_format('channels_last')


def CapsNet(gene, input_shape, n_class, routings):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    inputs = layers.Input(shape=input_shape)
    x = inputs

    print("### Input shape", )

    # if input_shape==(28, 28, 1):
    #     dataset="mnist"
    # elif input_shape==(32, 32, 3):
    #     dataset="cifar"
    # elif input_shape==(32, 32, 3) and gene[len(gene)-1][0]==2:
    #     dataset="cifar_resized"

    remaining=len(gene)-2
    conv_index=1
    caps_index=1
    deepcaps_index=1
    count=0
    xtra_skip=gene[len(gene)-2][0]
    print("xtra_skip is " +str(gene[len(gene)-2][0])+"\n")
    
    # Scan for type of layers
    #N_layers = int((len(gene)-2)/9) # -1 for xtraskip term
    conv=[]
    caps=[]
    dcaps=[]
    N_conv=0
    N_caps=0
    N_dcaps=0
    i=0
    prevlayertype=0
    for index in range(0,len(gene)):
        if gene[index][0]==0:
            N_conv = N_conv+1
            conv.append(i)
            i+=1
        elif gene[index][0]==1:
            type=1
            N_caps = N_caps+1
            caps.append(i)
            i+=1
        elif gene[index][0]==2:
            convert=1
            type=2
            N_dcaps = N_dcaps+1
            dcaps.append(i)
            i+=1



    for index in range(0,len(gene)):
    #for layer in gene:
        
        if len(gene[index])>1:

            # Convolutional layers
            # conv = [0, insize, inchannels, incapsules, kernsize, stride, outsize, outchannels, outcapsules]
            if gene[index][0]==0:
                if prevlayertype==2:
                    x=layers.Reshape((gene[index][1], gene[index][1], -1))(x)
                    print("ConvertToConv"+str(prevlayertype))

                x = layers.Conv2D(filters=gene[index][7]*gene[index][8], kernel_size=gene[index][4], strides=gene[index][5], padding='same', data_format="channels_last")(x)

                x = BatchNormalization()(x)
                print(x.get_shape().as_list())
                
                conv_index=conv_index+1
                remaining=remaining-1
                print("Added Conv%s_layer" % str(conv_index-1))
                count +=1
                if xtra_skip==count-1 or (xtra_skip==0 and count==1):
                    x1 = x
                    xtra_skip=-2 #flag==2 for inserted xtra skip
                prevlayertype=0

            # Primary Capsules layers
            # caps = [1, insize, inchannels, incapsules, kernsize, stride, outsize, outchannels, outcapsules]
            elif gene[index][0]==1 and remaining>1:
                x = PrimaryCap(x, dim_capsule=gene[index][8], n_channels=gene[index][7], kernel_size=gene[index][4], strides=gene[index][5], padding='same')  #CapsuleLayer(layer[7], layer[8])(x)
                print(x.get_shape().as_list())
            
                caps_index=caps_index+1
                remaining=remaining-1
                print("rem "+str(remaining))
                print("Added Caps%s_layer" % str(caps_index-1))
                veclen=gene[index][7]
                numout=gene[index][8]
                prevlayertype=1

            # Class Capsules layer
            # caps = [1, insize, inchannels, incapsules, kernsize, stride, outsize, outchannels, outcapsules]
            elif gene[index][0]==1 and remaining==1:
                print(x.get_shape().as_list())
                   
                x = layers.Reshape(target_shape=[-1, gene[index][2]])(x)  #[veclen*layer[1]*layer[1],  numout])(x) # added this to solve error instead of   Reshape(target_shape=[-1, layer[2]])(x) 
                print(x.get_shape().as_list()) 
                x = CapsuleLayer(gene[index][7], gene[index][8], 3, name='digitcaps')(x)
                print(x.get_shape().as_list()) 
                dim_capsule_out=gene[index][8]
                caps_index=caps_index+1
                remaining=remaining-1
                print(x.get_shape().as_list())
                print("Added ClassCaps_layer\n")

            # DeepCaps Cells
            # d_caps = [2, insize, inchannels, incapsules, kernsize, stride, outsize, outchannels, outcapsules]
            elif gene[index][0]==2 and remaining>1:
                if prevlayertype==0:
                    x = ConvertToCaps()(x)  
                    print("ConvertToCaps"+str(prevlayertype))

                print(x.get_shape().as_list())        
                x = Conv2DCaps(gene[index][7], gene[index][8], kernel_size=(gene[index][4], gene[index][4]), strides=(gene[index][5], gene[index][5]), r_num=1, b_alphas=[1, 1, 1])(x)
                deepcaps_index=deepcaps_index+1

                if remaining==2:
                    x_skip = ConvCapsuleLayer3D(kernel_size=3, num_capsule=gene[index][7], num_atoms=gene[index][8], strides=1, padding='same', routings=3)(x)
                    deepcaps_index=deepcaps_index+1
                else:
                    x_skip = Conv2DCaps(gene[index][7], gene[index][8], kernel_size=(gene[index][4], gene[index][4]), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(x)
                    deepcaps_index=deepcaps_index+1

                x = Conv2DCaps(gene[index][7], gene[index][8], kernel_size=(gene[index][4], gene[index][4]), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(x)
                deepcaps_index=deepcaps_index+1
                x = Conv2DCaps(gene[index][7], gene[index][8], kernel_size=(gene[index][4], gene[index][4]), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(x)
                x = layers.Add()([x, x_skip])
                count +=1 #CapsCell count
                print("Added CapsCell_layer\n")
                
                if xtra_skip-1==count-1: 
                    x1 = x
                    xtra_skip=-2 #flag==2 for inserted xtra skip
                elif remaining==1 and xtra_skip==-2:
                    x2 = x 
                elif remaining==2 and xtra_skip==-1:
                    x1 = x
                    x2 = x

                deepcaps_index=deepcaps_index+1
                remaining=remaining-1
                x2 = x
                prevlayertype=2                

            #FlattenCaps
            elif gene[index][0]==2 and remaining==1:
                print(x1.get_shape().as_list())
                print(x2.get_shape().as_list())
                
                flatCaps = FlattenCaps()
                xa = flatCaps(x2)
                flatCaps = FlattenCaps()
                xb = flatCaps(x1)
                x = layers.Concatenate(axis=-2)([xa, xb])
                print(xa.get_shape().as_list())
                x = DClassCaps(num_capsule=gene[index][7], dim_capsule=gene[index][8], routings=3, channels=0, name='digit_caps')(x) 
                print("Added FlattenCaps_layer\n")
                print(x.get_shape().as_list())
                dim_capsule_out=gene[index][8]

        else:
            break  

   
    digitcaps = x

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')

    # DeepCaps Decoder
    if input_shape == (32, 32, 3): # cifar10 resized
        decoder.add(layers.Dense(8*8*16, input_dim=dim_capsule_out*n_class, activation='relu')) #8*8*16
        decoder.add(layers.Reshape((8, 8, 16)))
        decoder.add(layers.BatchNormalization(momentum=0.8))
        decoder.add(layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), padding="same"))
        decoder.add(layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same"))
        decoder.add(layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding="same"))
        #decoder.add(layers.Conv2DTranspose(8, (3, 3), strides=(2, 2), padding="same"))
        decoder.add(layers.Conv2DTranspose(3, (3, 3), strides=(1, 1), padding="same"))
        decoder.add(layers.Activation("relu"))
        decoder.add(layers.Reshape(target_shape=(32, 32, 3), name='out_recon')) # 64, 64 in origial code
    elif input_shape == (64, 64, 3): # cifar 10 resized
        decoder.add(layers.Dense(8*8*16, input_dim=dim_capsule_out*n_class, activation='relu')) #8*8*16
        decoder.add(layers.Reshape((8, 8, 16)))
        decoder.add(layers.BatchNormalization(momentum=0.8))
        decoder.add(layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), padding="same"))
        decoder.add(layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same"))
        decoder.add(layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding="same"))
        decoder.add(layers.Conv2DTranspose(8, (3, 3), strides=(2, 2), padding="same"))
        decoder.add(layers.Conv2DTranspose(3, (3, 3), strides=(1, 1), padding="same"))
        decoder.add(layers.Activation("relu"))
        decoder.add(layers.Reshape(target_shape=(64, 64, 3), name='out_recon')) 
    elif input_shape == (28, 28, 1): # mnist
        decoder.add(layers.Dense(7*7*16, input_dim=dim_capsule_out*n_class, activation="relu")) #7*7*16
        decoder.add(layers.Reshape((7, 7, 16)))
        decoder.add(layers.BatchNormalization(momentum=0.8))
        decoder.add(layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), padding="same"))
        decoder.add(layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same"))
        decoder.add(layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding="same"))
        decoder.add(layers.Conv2DTranspose(1, (3, 3), strides=(1, 1), padding="same"))
        decoder.add(layers.Activation("relu"))
        decoder.add(layers.Reshape(target_shape=(28, 28, 1), name='out_recon'))
    elif input_shape == (56, 56, 1): # mnist resized
        decoder.add(layers.Dense(7*7*16, input_dim=dim_capsule_out*n_class, activation="relu")) #7*7*16
        decoder.add(layers.Reshape((7, 7, 16)))
        decoder.add(layers.BatchNormalization(momentum=0.8))
        decoder.add(layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), padding="same"))
        decoder.add(layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same"))
        decoder.add(layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding="same"))
        decoder.add(layers.Conv2DTranspose(4, (3, 3), strides=(1, 1), padding="same"))
        decoder.add(layers.Activation("relu"))
        decoder.add(layers.Reshape(target_shape=(56, 56, 1), name='out_recon'))
    
    elif input_shape == (56, 56, 3): # mnist resized
        decoder.add(layers.Dense(7*7*16, input_dim=dim_capsule_out*n_class, activation="relu")) #7*7*16
        decoder.add(layers.Reshape((7, 7, 16)))
        decoder.add(layers.BatchNormalization(momentum=0.8))
        decoder.add(layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), padding="same"))
        decoder.add(layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same"))
        decoder.add(layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding="same"))
        decoder.add(layers.Conv2DTranspose(12, (3, 3), strides=(1, 1), padding="same"))
        decoder.add(layers.Activation("relu"))
        decoder.summary()
        decoder.add(layers.Reshape(target_shape=(56, 56, 3), name='out_recon'))


    elif input_shape == (64, 64, 1): # mnist resized for deepcaps (64x64 inputs)
        decoder.add(layers.Dense(8*8*16, input_dim=dim_capsule_out*n_class, activation='relu')) #8*8*16
        decoder.add(layers.Reshape((8, 8, 16)))
        decoder.add(layers.BatchNormalization(momentum=0.8))
        decoder.add(layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), padding="same"))
        decoder.add(layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same"))
        decoder.add(layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding="same"))
        decoder.add(layers.Conv2DTranspose(8, (3, 3), strides=(2, 2), padding="same"))
        decoder.add(layers.Conv2DTranspose(1, (3, 3), strides=(1, 1), padding="same"))
        decoder.add(layers.Activation("relu"))
        decoder.add(layers.Reshape(target_shape=(64, 64, 1), name='out_recon')) 
    else:
        raise NotImplementedError(f"Unknown decoder for shape {input_shape}")
    decoder.summary()

    # Models for training and evaluation (prediction)
    train_model = models.Model([inputs, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(inputs, [out_caps, decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=(n_class, dim_capsule_out))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([inputs, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model



