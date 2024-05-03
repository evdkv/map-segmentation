'''
File containing the FCN-32 model architecture

Authors: Sophie Zhao, Tolya Evdokimov

Adapted from: 
https://github.com/divamgupta/image-segmentation-keras/blob/master/keras_segmentation/models/fcn.py
'''

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, \
Conv2DTranspose, ZeroPadding2D, BatchNormalization, Activation

IMAGE_ORDERING = 'channels_last'

def encoder(input_height, input_width, channels):
    '''
    The encoder for the FCN-32 model

    Parameters:
        input_height (int): The height of the input images
        input_width (int): The width of the input images
        channels (int): The number of channels in the input images
    
    Returns:
        Input: The input layer
        list: The list of layers from the encoder
    '''

    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2

    img_input = Input(shape=(input_height, input_width, channels), name='input_1')

    x = img_input
    levels = []

    # Block 1
    x = (ZeroPadding2D((pad, pad), data_format=IMAGE_ORDERING))(x)
    x = (Conv2D(filter_size, (kernel, kernel), data_format=IMAGE_ORDERING, padding='valid'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((pool_size, pool_size), data_format=IMAGE_ORDERING))(x)
    levels.append(x)

    # Block 2
    x = (ZeroPadding2D((pad, pad), data_format=IMAGE_ORDERING))(x)
    x = (Conv2D(128, (kernel, kernel), data_format=IMAGE_ORDERING, padding='valid'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((pool_size, pool_size), data_format=IMAGE_ORDERING))(x)
    levels.append(x)

    # Block 3
    for _ in range(3):
        x = (ZeroPadding2D((pad, pad), data_format=IMAGE_ORDERING))(x)
        x = (Conv2D(256, (kernel, kernel), data_format=IMAGE_ORDERING, padding='valid'))(x)
        x = (BatchNormalization())(x)
        x = (Activation('relu'))(x)
        x = (MaxPooling2D((pool_size, pool_size), data_format=IMAGE_ORDERING))(x)
        levels.append(x)

    return img_input, levels

def fcn_32(n_classes, input_height, input_width, channels, encoder=encoder):
    '''
    The FCN-32 model

    Parameters:
        n_classes (int): The number of classes
        input_height (int): The height of the input images
        input_width (int): The width of the input images
        channels (int): The number of channels in the input images
        encoder (function): The encoder function
    
    Returns:
        Model: The FCN-32 model
    '''

    img_input, levels = encoder(input_height=input_height,  input_width=input_width, channels=channels)

    # Extract skip connections
    [f1, f2, f3, f4, f5] = levels

    # Add convolutional layers
    x = f5

    x = (Conv2D(4096, (7, 7), activation='relu', padding='same', data_format=IMAGE_ORDERING))(x)
    x = Dropout(0.5)(x)
    x = (Conv2D(4096, (1, 1), activation='relu', padding='same', data_format=IMAGE_ORDERING))(x)
    x = Dropout(0.5)(x)

    x = (Conv2D(n_classes,  (1, 1), kernel_initializer='he_normal',data_format=IMAGE_ORDERING , name="seg_feats" ))(x)
    x = Conv2DTranspose(n_classes, kernel_size=(32, 32), trides=(32, 32), use_bias=False,  data_format=IMAGE_ORDERING)(x)

    x = (Activation('softmax', name="conv2d_18"))(x)

    model = Model(img_input, x)

    return model