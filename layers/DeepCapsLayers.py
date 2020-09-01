# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:06:02 2020
@author: MA
"""
import tensorflow as tf
import numpy as np
from config import cfg
import keras as k
from keras.utils import conv_utils
from keras.layers import Input, InputSpec
from keras import backend as K
epsilon = 1e-9

class ConvertToCaps(object):

    def __init__(self, **kwargs):
        super(ConvertToCaps, self).__init__(**kwargs)
        # self.input_spec = InputSpec(min_ndim=2)

#    def compute_output_shape(self, input):
#        input_shape=input.get_shape()
#
#        output_shape = list(input_shape)
#        output_shape.insert(1 if cf else len(output_shape), 1)
#        return tuple(output_shape)

    def call(self, input):
        input_shape=input.get_shape()
        #return K.expand_dims(inputs, 1 if cf else -1)
        return tf.reshape(input, shape = (input_shape[0:3], 1))

    def get_config(self):
        config = {
            'input_spec': 5
        }
        base_config = super(ConvertToCaps, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Conv2DCaps(object):

    def __init__(self, ch_j, n_j,
                 kernel_size=(3,3),
                 stride=(1,1),
                 r_num=1,
                 b_alphas=[8, 8, 8],
                 padding='SAME',
                 data_format='channels_last',
                 dilation_rate=(1, 1),
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        super(Conv2DCaps, self).__init__(**kwargs)
        rank = 2
        self.ch_j = ch_j  # Number of capsules in layer J
        self.n_j = n_j  # Number of neurons in a capsule in J
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.stride = conv_utils.normalize_tuple(stride, rank, 'stride')
        self.r_num = r_num
        self.b_alphas = b_alphas
        self.padding = conv_utils.normalize_padding(padding)
        try:
            self.data_format = k.backend.common.normalize_data_format(data_format)
        except AttributeError: # different version of Keras
            self.data_format = conv_utils.normalize_data_format(data_format)

        self.dilation_rate = (1, 1)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.input_spec = tf.keras.layers.InputSpec(ndim=rank + 3)


   #def __call__(self, input, input_shape):

#    def build(self, input_shape):
    def __call__(self, input):
        #assert input.get_shape() == [cfg.batch_size, self.in_size, self.in_size, self.prev_num_outputs_veclen]
        input_shape=input.get_shape()
        print(input_shape)

        self.h_i, self.w_i, self.ch_i, self.n_i = input_shape[1:5]

        self.h_j, self.w_j = [conv_utils.conv_output_length(input_shape[i + 1],
                                                            self.kernel_size[i],
                                                            padding=self.padding,
                                                            stride=self.stride[i],
                                                            dilation=self.dilation_rate[i]) for i in (0, 1)]

        self.ah_j, self.aw_j = [conv_utils.conv_output_length(input_shape[i + 1],
                                                              self.kernel_size[i],
                                                              padding=self.padding,
                                                              stride=1,
                                                              dilation=self.dilation_rate[i]) for i in (0, 1)]

        self.w_shape = self.kernel_size + (self.ch_i, self.n_i,
                                           self.ch_j, self.n_j)

#        self.w = self.add_weight(shape=self.w_shape,
#                                 initializer=self.kernel_initializer,
#                                 name='kernel',
#                                 regularizer=self.kernel_regularizer,
#                                 constraint=self.kernel_constraint)
        self.w = tf.compat.v1.get_variable('Weight', shape=self.w_shape, dtype=tf.float32,
                                           initializer=tf.random_normal_initializer)#self.kernel_initializer)

        #self.built = True


        if self.r_num == 1:
            # if there is no routing (and this is so when r_num is 1 and all c are equal)
            # then this is a common convolution
            outputs = tf.keras.backend.conv2d(tf.reshape(input, shape = (-1, self.h_i, self.w_i, self.ch_i * self.n_i)),
                                               tf.reshape(self.w, shape = (self.kernel_size + (self.ch_i * self.n_i, self.ch_j * self.n_j))),
                                               data_format='channels_last',
                                               strides=self.stride,
                                               padding=self.padding,
                                               dilation_rate=self.dilation_rate)

            outputs = tf.squeeze(tf.reshape(outputs, shape = ((-1, self.h_j, self.w_j, self.ch_j, self.n_j))))

        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.h_j, self.w_j, self.ch_j, self.n_j)

    def get_config(self):
        config = {
            'ch_j': self.ch_j,
            'n_j': self.n_j,
            'kernel_size': self.kernel_size,
            'strides': self.stride,
            'b_alphas': self.b_alphas,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'kernel_initializer': tf.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': tf.regularizers.serialize(self.kernel_regularizer),
            'activity_regularizer': tf.regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': tf.constraints.serialize(self.kernel_constraint)
        }
        base_config = super(Conv2DCaps, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



    

class ConvCapsuleLayer3D(object):
    
    def __init__(self, kernel_size, num_capsule, num_atoms, strides=1, padding='same', routings=3, kernel_initializer='he_normal', **kwargs):
        self.kernel_size = kernel_size
        self.num_capsule = num_capsule
        self.num_atoms = num_atoms
        self.strides = strides
        self.padding = padding
        self.routings = routings
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)

    def __call__(self, input, training=None):
        
        input_shape=input.get_shape()
        print(input_shape)
        assert len(input_shape) == 5
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.input_num_capsule = input_shape[3]
        self.input_num_atoms = input_shape[4]

        # Transform matrix
        self.W = tf.compat.v1.get_variable('W', shape=[self.input_num_atoms, self.kernel_size, self.kernel_size, 1, self.num_capsule * self.num_atoms], dtype=tf.float32,
                                   initializer=tf.random_normal_initializer)#self.kernel_initializer)

        self.b = tf.compat.v1.get_variable('b', shape=[self.num_capsule, self.num_atoms, 1, 1], dtype=tf.float32,
                                   initializer=tf.compat.v1.initializers.constant(0.1))

        input_transposed = tf.transpose(input, [0, 3, 4, 1, 2])
        input_shape = tf.shape(input_transposed)
        input_tensor_reshaped = tf.reshape(input, shape=(input_shape[0], 1, self.input_num_capsule * self.input_num_atoms, self.input_height, self.input_width))

        input_tensor_reshaped.set_shape((None, 1, self.input_num_capsule * self.input_num_atoms, self.input_height, self.input_width))
        
        conv = K.conv3d(input_tensor_reshaped, self.W, strides=(self.input_num_atoms, self.strides, self.strides), padding='same', data_format='channels_first')
        #conv = tf.nn.conv3d(input_tensor_reshaped, self.W, strides=(1, self.input_num_atoms, self.strides, self.strides, 1), padding=self.padding, data_format='NCDHW')
        #print("conv"+str(conv.get_shape().as_list())+"\n\n")
        votes_shape = tf.shape(conv)
        #print(conv.get_shape().as_list())
        _, _, _, conv_height, conv_width = conv.get_shape()
        conv = tf.transpose(conv, [0, 2, 1, 3, 4])
        #print(conv.get_shape().as_list())
        #print([input_shape[0], self.input_num_capsule, self.num_capsule, self.num_atoms, votes_shape[3], votes_shape[4]])
        votes = tf.reshape(conv, shape=(input_shape[0], self.input_num_capsule, self.num_capsule, self.num_atoms, votes_shape[3], votes_shape[4]))
        votes.set_shape((None, self.input_num_capsule, self.num_capsule, self.num_atoms, conv_height.value, conv_width.value))

        logit_shape = tf.stack([input_shape[0], self.input_num_capsule, self.num_capsule, votes_shape[3], votes_shape[4]])
        biases_replicated = tf.tile(self.b, [1, 1, conv_height.value, conv_width.value])

        activations = update_routing(
            votes=votes,
            biases=biases_replicated,
            logit_shape=logit_shape,
            num_dims=6,
            input_dim=self.input_num_capsule,
            output_dim=self.num_capsule,
            num_routing=self.routings)

        a2 = tf.transpose(activations, [0, 3, 4, 1, 2])
        return a2

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-2]
        new_space = []
        for i in range(len(space)):
            new_dim = tf.conv_output_length(space[i], self.kernel_size, padding=self.padding, stride=self.strides, dilation=1)
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_capsule, self.num_atoms)

    def get_config(self):
        config = {
            'kernel_size': self.kernel_size,
            'num_capsule': self.num_capsule,
            'num_atoms': self.num_atoms,
            'strides': self.strides,
            'padding': self.padding,
            'routings': self.routings,
            'kernel_initializer': tf.compat.v1.initializers.serialize(self.kernel_initializer)
        }
        base_config = super(ConvCapsuleLayer3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
  
    
    
class FlattenCaps(object):
    
    def __init__(self, in_size):
        self.in_size=in_size
        self.input_spec = InputSpec(min_ndim=4)

    def compute_output_shape(self, input):
        #input_shape=[self.in_size, self.in_size]
        input_shape=input.get_shape().as_list()
        if not all(input_shape[1:]):
            raise ValueError('The shape of the input to "FlattenCaps" '
                             'is not fully defined '
                             '(got ' + str(input_shape[1:]) + '. '
                             'Make sure to pass a complete "input_shape" '
                             'or "batch_input_shape" argument to the first '
                             'layer in your model.')
        return (self.in_size, np.prod(input_shape[1:-1]), self.in_size)

    def __call__(self, input):
        shape = input.get_shape()
        FlatCap = K.reshape(input, (-1, np.prod(shape[1:-1]), shape[-1])) #tf.reshape(input, shape=(cfg.batch_size, np.prod(shape[1:-1]), shape[-1]))
        print("FlatCaps: "+str(FlatCap.get_shape().as_list()))
        return FlatCap
        

def update_routing(votes, biases, logit_shape, num_dims, input_dim, output_dim,
                   num_routing):
    if num_dims == 6:
        votes_t_shape = [3, 0, 1, 2, 4, 5]
        r_t_shape = [1, 2, 3, 0, 4, 5]
    elif num_dims == 4:
        votes_t_shape = [3, 0, 1, 2]
        r_t_shape = [1, 2, 3, 0]
    else:
        raise NotImplementedError('Not implemented')

    votes_trans = tf.transpose(votes, votes_t_shape)
    _, _, _, height, width, caps = votes_trans.get_shape()

    def _body(i, logits, activations):
        """Routing while loop."""
        # route: [batch, input_dim, output_dim, ...]
        a,b,c,d,e = logits.get_shape()
        a = logit_shape[0]
        b = logit_shape[1]
        c = logit_shape[2]
        d = logit_shape[3]
        e = logit_shape[4]
        print(logit_shape)
        logit_temp = tf.reshape(logits, [a,b,-1])
        route_temp = tf.nn.softmax(logit_temp, axis=-1)
        route = tf.reshape(route_temp, [a, b, c, d, e])
        preactivate_unrolled = route * votes_trans
        preact_trans = tf.transpose(preactivate_unrolled, r_t_shape)
        preactivate = tf.reduce_sum(preact_trans, axis=1) + biases
        # activation = _squash(preactivate)
        activation = squash(preactivate)
        activations = activations.write(i, activation)

        act_3d = tf.expand_dims(activation, 1)
        tile_shape = np.ones(num_dims, dtype=np.int32).tolist()
        tile_shape[1] = input_dim
        act_replicated = tf.tile(act_3d, tile_shape)
        distances = tf.reduce_sum(votes * act_replicated, axis=3)
        logits += distances
        return (i + 1, logits, activations)

    activations = tf.TensorArray(
        dtype=tf.float32, size=num_routing, clear_after_read=False)
    logits = tf.fill(logit_shape, 0.0)

    i = tf.constant(0, dtype=tf.int32)
    _, logits, activations = tf.while_loop(
        lambda i, logits, activations: i < num_routing,
        _body,
        loop_vars=[i, logits, activations],
        swap_memory=True)
    a = tf.cast(activations.read(num_routing - 1), dtype='float32')
    return a


class DClassCaps(object):
    def __init__(self, num_capsule, dim_capsule, channels, routings=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.channels = channels
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)

    def __call__(self, input, training=None):
        input_shape=input.get_shape().as_list()
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        if(self.channels != 0):
            assert int(self.input_num_capsule / self.channels) / (self.input_num_capsule / self.channels) == 1, "error"
            self.W = tf.compat.v1.get_variable('W', shape=[self.num_capsule, self.channels,
                                                           self.dim_capsule, self.input_dim_capsule], dtype=tf.float32,
                                                           initializer=tf.random_normal_initializer)#self.kernel_initializer)
    
            self.B = tf.compat.v1.get_variable('B', shape=[self.num_capsule, self.dim_capsule], dtype=tf.float32,
                                       initializer=self.kernel_initializer)

        else:

            self.W = tf.compat.v1.get_variable('W', shape=[self.num_capsule, self.input_num_capsule,
                                                           self.dim_capsule, self.input_dim_capsule], dtype=tf.float32,
                                                           initializer=tf.random_normal_initializer)#self.kernel_initializer)
    
            self.B = tf.compat.v1.get_variable('B', shape=[self.num_capsule, self.dim_capsule], dtype=tf.float32,
                                       initializer=self.kernel_initializer)

        # inputs.shape=[None, input_num_capsule, input_dim_capsule]
        # inputs_expand.shape=[None, 1, input_num_capsule, input_dim_capsule]
        inputs_expand = K.expand_dims(input, 1)

        # Replicate num_capsule dimension to prepare being multiplied by W
        # inputs_tiled.shape=[None, num_capsule, input_num_capsule, input_dim_capsule]
        inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])

        if(self.channels != 0):
            W2 = K.repeat_elements(self.W, int(self.input_num_capsule / self.channels), 1)
        else:
            W2 = self.W
        # Compute `inputs * W` by scanning inputs_tiled on dimension 0.
        # x.shape=[num_capsule, input_num_capsule, input_dim_capsule]
        # W.shape=[num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
        # Regard the first two dimensions as `batch` dimension,
        # then matmul: [input_dim_capsule] x [dim_capsule, input_dim_capsule]^T -> [dim_capsule].
        # inputs_hat.shape = [None, num_capsule, input_num_capsule, dim_capsule]
        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, W2, [2, 3]), elems=inputs_tiled)

        # Begin: Routing algorithm ---------------------------------------------------------------------#
        # The prior for coupling coefficient, initialized as zeros.
        # b.shape = [None, self.num_capsule, self.input_num_capsule].
        b = tf.zeros(shape=[K.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])

        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            # c.shape=[batch_size, num_capsule, input_num_capsule]
            c = tf.nn.softmax(b, dim=1)

            # c.shape =  [batch_size, num_capsule, input_num_capsule]
            # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
            # The first two dimensions as `batch` dimension,
            # then matmal: [input_num_capsule] x [input_num_capsule, dim_capsule] -> [dim_capsule].
            # outputs.shape=[None, num_capsule, dim_capsule]
            outputs = squash(K.batch_dot(c, inputs_hat, [2, 2]) + self.B)  # [None, 10, 16]

            if i < self.routings - 1:
                # outputs.shape =  [None, num_capsule, dim_capsule]
                # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
                # The first two dimensions as `batch` dimension,
                # then matmal: [dim_capsule] x [input_num_capsule, dim_capsule]^T -> [input_num_capsule].
                # b.shape=[batch_size, num_capsule, input_num_capsule]
                b += K.batch_dot(outputs, inputs_hat, [2, 3])
        # End: Routing algorithm -----------------------------------------------------------------------#

        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])



class Mask_CID(object):

    def __call__(self, input, **kwargs):
        if isinstance(input, list):  # true label is provided with shape = [None, n_classes], i.e. one-hot code.
            assert len(input) == 2
            input, a = input
            mask = tf.keras.backend.argmax(a, 1)
        else:  # if no true label, mask by the max length of capsules. Mainly used for prediction
            # compute lengths of capsules
            x = tf.keras.backend.sqrt(tf.keras.backend.sum(tf.keras.backend.square(input), -1))
            # generate the mask which is a one-hot code.
            # mask.shape=[None, n_classes]=[None, num_capsule]
            mask = tf.keras.backend.argmax(x, 1)

        increasing = tf.range(start=0, limit=tf.shape(input)[0], delta=1)
        m = tf.stack([increasing, tf.cast(mask, tf.int32)], axis=1)
        # inputs.shape=[None, num_capsule, dim_capsule]
        # mask.shape=[None, num_capsule]
        # masked.shape=[None, num_capsule * dim_capsule]
        # x1 = tf.transpose(inputs, (0))
        masked = tf.gather_nd(input, m)

        return masked

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape[0], tuple):  # true label provided
            return tuple([None, input_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_shape[2]])
        
def squash(vector):
    '''Squashing function
    Args:
        vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
    Returns:
        A tensor with the same shape as vector but squashed in 'vec_len' dimension.
    '''
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return vec_squashed
