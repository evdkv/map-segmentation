from keras.models import *
from keras.layers import *

from keras.utils import plot_model

IMAGE_ORDERING = 'channels_last'

def encoder(input_height, input_width, channels):

    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2

    img_input = Input(shape=(input_height, input_width, channels))

    x = img_input
    levels = []

    x = (ZeroPadding2D((pad, pad), data_format=IMAGE_ORDERING))(x)
    x = (Conv2D(filter_size, (kernel, kernel),
                data_format=IMAGE_ORDERING, padding='valid'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((pool_size, pool_size), data_format=IMAGE_ORDERING))(x)
    levels.append(x)

    x = (ZeroPadding2D((pad, pad), data_format=IMAGE_ORDERING))(x)
    x = (Conv2D(128, (kernel, kernel), data_format=IMAGE_ORDERING,
         padding='valid'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((pool_size, pool_size), data_format=IMAGE_ORDERING))(x)
    levels.append(x)

    for _ in range(3):
        x = (ZeroPadding2D((pad, pad), data_format=IMAGE_ORDERING))(x)
        x = (Conv2D(256, (kernel, kernel),
                    data_format=IMAGE_ORDERING, padding='valid'))(x)
        x = (BatchNormalization())(x)
        x = (Activation('relu'))(x)
        x = (MaxPooling2D((pool_size, pool_size),
             data_format=IMAGE_ORDERING))(x)
        levels.append(x)

    return img_input, levels

def crop(o1, o2, i):
    o_shape2 = Model(i, o2).output_shape

    output_height2 = o_shape2[1]
    output_width2 = o_shape2[2]

    o_shape1 = Model(i, o1).output_shape

    output_height1 = o_shape1[1]
    output_width1 = o_shape1[2]

    cx = abs(output_width1 - output_width2)
    cy = abs(output_height2 - output_height1)

    if output_width1 > output_width2:
        o1 = Cropping2D(cropping=((0, 0),  (0, cx)),
                        data_format=IMAGE_ORDERING)(o1)
    else:
        o2 = Cropping2D(cropping=((0, 0),  (0, cx)),
                        data_format=IMAGE_ORDERING)(o2)

    if output_height1 > output_height2:
        o1 = Cropping2D(cropping=((0, cy),  (0, 0)),
                        data_format=IMAGE_ORDERING)(o1)
    else:
        o2 = Cropping2D(cropping=((0, cy),  (0, 0)),
                        data_format=IMAGE_ORDERING)(o2)

    return o1, o2

def fcn_32(n_classes, input_height, input_width, channels, encoder=encoder):

    img_input, levels = encoder(
        input_height=input_height,  input_width=input_width, channels=channels)
    [f1, f2, f3, f4, f5] = levels

    o = f5

    o = (Conv2D(4096, (7, 7), activation='relu',
                padding='same', data_format=IMAGE_ORDERING))(o)
    o = Dropout(0.5)(o)
    o = (Conv2D(4096, (1, 1), activation='relu',
                padding='same', data_format=IMAGE_ORDERING))(o)
    o = Dropout(0.5)(o)

    o = (Conv2D(n_classes,  (1, 1), kernel_initializer='he_normal',
                data_format=IMAGE_ORDERING , name="seg_feats" ))(o)
    o = Conv2DTranspose(n_classes, kernel_size=(32, 32),  strides=(
        32, 32), use_bias=False,  data_format=IMAGE_ORDERING)(o)
    o = (Activation('softmax', name="conv2d_18"))(o)

    model = Model(img_input, o)

    return model


def fcn_8(n_classes, input_height, input_width, channels, encoder=encoder):
    img_input, levels = encoder(
    input_height=input_height,  input_width=input_width, channels=channels)
    [f1, f2, f3, f4, f5] = levels

    o = f5

    o = (Conv2D(4096, (7, 7), activation='relu',
                padding='same', data_format=IMAGE_ORDERING))(o)
    o = Dropout(0.5)(o)
    o = (Conv2D(4096, (1, 1), activation='relu',
                padding='same', data_format=IMAGE_ORDERING))(o)
    o = Dropout(0.5)(o)

    o = (Conv2D(n_classes,  (1, 1), kernel_initializer='he_normal',
                data_format=IMAGE_ORDERING))(o)
    o = Conv2DTranspose(n_classes, kernel_size=(4, 4),  strides=(
        2, 2), use_bias=False, data_format=IMAGE_ORDERING)(o)

    o2 = f4
    o2 = (Conv2D(n_classes,  (1, 1), kernel_initializer='he_normal',
                 data_format=IMAGE_ORDERING))(o2)

    o, o2 = crop(o, o2, img_input)

    o = Add()([o, o2])

    o = Conv2DTranspose(n_classes, kernel_size=(4, 4),  strides=(
        2, 2), use_bias=False, data_format=IMAGE_ORDERING)(o)
    o2 = f3
    o2 = (Conv2D(n_classes,  (1, 1), kernel_initializer='he_normal',
                 data_format=IMAGE_ORDERING))(o2)
    o2, o = crop(o2, o, img_input)
    o = Add( name="seg_feats" )([o2, o])

    o = Conv2DTranspose(n_classes, kernel_size=(8, 8),  strides=(
        8, 8), use_bias=False, data_format=IMAGE_ORDERING)(o)

    o = (Activation('softmax', name="conv2d_18"))(o)

    return Model(img_input, o)