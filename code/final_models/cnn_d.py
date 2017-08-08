

# -*- coding: utf-8 -*-
"""VGG16 model for Keras.
# The CNN is modified from the VGG16 network implemented at
# https://github.com/fchollet/keras/blob/master/keras/applications/vgg16.py
# Reference
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
"""
from __future__ import print_function
from __future__ import absolute_import

import warnings

from keras.models import *
from keras.layers import *
from keras.applications.imagenet_utils import _obtain_input_shape
import keras.backend as K
from keras.utils.data_utils import *



def network(include_top=True, weights=None,
          input_tensor=None, input_shape=None,
          pooling=None,
          classes=5):
    """
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 input channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
#     if weights not in {'imagenet', None}:
#         raise ValueError('The `weights` argument should be either '
#                          '`None` (random initialization) or `imagenet` '
#                          '(pre-training on ImageNet).')

#     if weights == 'imagenet' and include_top and classes != 1000:
#         raise ValueError('If using `weights` as imagenet with `include_top`'
#                          ' as true, `classes` should be 1000')
    # Determine proper input shape
#     input_shape = _obtain_input_shape(input_shape,
#                                       default_size=224,
#                                       min_size=48,
#                                       data_format=K.image_data_format(),
#                                       include_top=include_top)
    input_shape = (196,196,3)
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1 (64)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

#     # Block 4
    # x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    # x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    # x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

#     # Block 5
#     x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
#     x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
#     x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
#     x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(64, activation='relu', name='fc1')(x)
        # x = Dense(32, activation='relu', name='fc2')(x)

        x = Dropout(0.3)(x)
        
#         x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='sigmoid', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='cnn_d')


    return model