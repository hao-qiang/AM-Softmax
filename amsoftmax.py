# -*- coding: utf-8 -*-
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Dense, Activation,BatchNormalization
from keras.layers import activations, initializers, regularizers, constraints, Lambda
from keras.engine import InputSpec
import tensorflow as tf
import numpy as np

class AMSoftmax(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(AMSoftmax, self).__init__(**kwargs)


    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(AMSoftmax, self).build(input_shape)


    def call(self, inputs):
        inputs = tf.nn.l2_normalize(inputs, dim=1)  # input_l2norm
        self.kernel = tf.nn.l2_normalize(self.kernel, dim=0)   # W_l2norm

        cosine = K.dot(inputs, self.kernel)  # cos = input_l2norm * W_l2norm
        return cosine


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)



def amsoftmax_loss(y_true, y_pred):
    scale=30.0
    margin=0.35

    label = tf.reshape(tf.argmax(y_true, axis=-1), shape=(-1,1))
    label = tf.cast(label, dtype=tf.int32) # y
    batch_range = tf.reshape(tf.range(tf.shape(y_pred)[0]),shape=(-1,1)) # 0~batchsize-1
    indices_of_groundtruth = tf.concat([batch_range, tf.reshape(label,shape=(-1,1))], axis=1) # 2columns vector, 0~batchsize-1 and label
    groundtruth_score = tf.gather_nd(y_pred, indices_of_groundtruth) # score of groundtruth

    m = tf.constant(margin,name='m')
    s = tf.constant(scale,name='s')
        
    added_margin = tf.cast(tf.greater(groundtruth_score,m),dtype=tf.float32)*m # if groundtruth_score>m, groundtruth_score-m
    added_margin = tf.reshape(added_margin,shape=(-1,1))
    added_embeddingFeature = tf.subtract(y_pred, y_true*added_margin)*s # s(cos_theta_yi-m), s(cos_theta_j)

    cross_ent = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=added_embeddingFeature)
    loss = tf.reduce_mean(cross_ent)
    return loss
