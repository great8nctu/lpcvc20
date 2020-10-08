from .utils import set_model_regularization

import numpy as np
import tensorflow as tf

he_normal_fan_out = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_out')

import math

class ConvLayer():
    def __init__(
            self,
            out_channels,
            kernel=3,
            stride=1,
            bias=False):
        super(ConvLayer, self).__init__()

        self.conv = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=kernel,
            strides=stride,
            padding="same",
            use_bias=bias,
            kernel_initializer=he_normal_fan_out)
        self.norm = tf.keras.layers.BatchNormalization(
            fused=False, scale=True, epsilon=1e-5, momentum=0.9)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.norm(x)
        return x


class DWConvLayer():
    def __init__(
            self, 
            kernel, 
            stride=1, 
            bias=False):
        super().__init__()
        
        self.dwconv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=kernel,
            strides=stride,
            padding="same",
            use_bias=bias,
            depthwise_initializer=he_normal_fan_out)
        self.norm = tf.keras.layers.BatchNormalization(
            fused=False, scale=True, epsilon=1e-5, momentum=0.9)

    def call(self, inputs):
        x = self.dwconv(inputs)
        x = self.norm(x)
        return x


class Hswish():
    def __init__(self):
        super(Hswish, self).__init__()

    def call(self, inputs):
        return tf.keras.layers.ReLU(max_value=6)(inputs)


class Hsig():
    def __init__(self):
        super(Hsig, self).__init__()

    def call(self, inputs):
        return tf.keras.layers.ReLU(max_value=6)(inputs + 3) * (1 / 6)


class SEL():
    def __init__(self, mid, oc):
        super(SEL, self).__init__()
        
        self.fc = tf.keras.layers.Conv2D(
            oc,
            1,
            1,
            use_bias=True,
            padding="same",
            kernel_initializer=he_normal_fan_out)
        self.relu = tf.keras.layers.ReLU(max_value=6)
        self.fc1 = tf.keras.layers.Conv2D(
            mid,
            1,
            1,
            use_bias=True,
            padding="same",
            kernel_initializer=he_normal_fan_out)
        self.hg = Hsig()

    def call(self, x):

        inputs = x
        x = tf.keras.layers.AveragePooling2D((x.shape[1], x.shape[1]))(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.hg.call(x)
        x = tf.keras.layers.Multiply()([inputs, x])
        return x



class MBI():
    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=3,
            strides=1,
            expand_ratio=2,
            use_se=False,
            shortcut = True):
        super(MBI, self).__init__()
        
        self.expand_ratio = expand_ratio
        self.shortcut = shortcut
        
        oc = in_ch*expand_ratio/4/8
        oc = math.ceil(oc)*8

        feature_dim = in_ch * self.expand_ratio

        if expand_ratio != 1:
            self.inv_bot = ([
                ConvLayer(
                    feature_dim, 
                    kernel=1, 
                    stride=1, 
                    bias=False).call,
                tf.keras.layers.ReLU(max_value=6)])

        self.depth_con = [
            DWConvLayer(
                kernel_size,
                stride=strides,
                bias=False).call,
            tf.keras.layers.ReLU(max_value=6)]

        if use_se:
            self.depth_con.append(SEL(feature_dim, oc).call)

        self.point_linear = [
            ConvLayer(
                out_ch,
                kernel=1,
                stride=1,
                bias=False).call]

    def call(self, x):
        o = x
        if self.expand_ratio != 1:
            for i in self.inv_bot:
                o = i(o)
                
        for i in self.depth_con:
            o = i(o)

        for i in self.point_linear:
            o = i(o)
            
        if self.shortcut:
            o = tf.keras.layers.Add()([o, x])

        return o


def fpnet():
    x = tf.keras.Input((224, 224, 3))
    w = ConvLayer(16, kernel=3, stride=2, bias=False).call(x)
    w = Hswish().call(w)

    #            [inp, oup, k, s, e, se, shortcut]
    block_list =[[16 , 16 , 3, 1, 1, 0, 1],
    
                 [16 , 24 , 3, 2, 3, 0, 0],
                 [24 , 24 , 3, 1, 3, 0, 1],
                 
                 [24 , 40 , 3, 2, 3, 1, 0],
                 [40 , 40 , 3, 1, 3, 1, 1],
                 [40 , 40 , 3, 1, 3, 1, 1],
                 
                 [40 , 80 , 7, 2, 3, 0, 0],
                 [80 , 80 , 3, 1, 3, 0, 1],
                 [80 , 80 , 3, 1, 3, 0, 1],
                 [80 , 80 , 3, 1, 3, 0, 1],
                 
                 [80 , 112, 3, 1, 7, 1, 0],
                 [112, 112, 3, 1, 6, 1, 1],
                 [112, 112, 3, 1, 6, 1, 1],
                 
                 [112, 160, 7, 2, 7, 1, 0],
                 [160, 160, 3, 1, 7, 1, 1],
                 [160, 160, 3, 1, 7, 1, 1],
                 [160, 160, 3, 1, 7, 1, 1]]
     
    for i in block_list:
        w = MBI(i[0], i[1], i[2], i[3], i[4], i[5], i[6]).call(w)

    w = tf.keras.layers.Conv2D(
        960,
        1,
        1,
        use_bias=False,
        padding="same",
        kernel_initializer=he_normal_fan_out)(w)
    w = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9)(w)

    w = Hswish().call(w)
    w = tf.keras.layers.AveragePooling2D((7, 7))(w)

    w = tf.keras.layers.Conv2D(
        1280,
        1,
        1,
        use_bias=True,
        padding="same",
        kernel_initializer=he_normal_fan_out)(w)
    w = Hswish().call(w)
    w = tf.keras.layers.Flatten()(w)
    w = tf.keras.layers.Dropout(0.2)(w)
    w = tf.keras.layers.Dense(1001, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))(w)
    w = tf.keras.layers.Softmax()(w)
    outputs = (w)
    model = tf.keras.Model(x, outputs)
    return model


def get_model():
    model = fpnet()
    weight_decay = 4e-5
    model = set_model_regularization(model, tf.keras.regularizers.l2(
        weight_decay / 2.0), layers=None, bias_term=False, bn_term=False, depthwise_term=True)
    momentum = 0.9
    label_smoothing = 0.1
    model.compile(
        optimizer=tf.keras.optimizers.SGD(
            momentum=momentum,
            nesterov=True),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=['categorical_accuracy'])
    return model
