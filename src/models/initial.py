import tensorflow as tf

def unet(num_classes):
    inputs = tf.keras.layers.Input((512, 512, 3))
    x = inputs

    # Downsampling
    skips = []
    for filters in [64, 128, 256, 512]:
        x = tf.keras.layers.Conv2D(filters, 3, strides=2, padding='same', activation='relu')(x)
        skips.append(x)

    # Bottleneck
    x = tf.keras.layers.Conv2D(1024, 3, strides=2, padding='same', activation='relu')(x)

    # Upsampling
    skips = skips[::-1]
    for filters in [512, 256, 128, 64]:
        x = tf.keras.layers.Conv2DTranspose(filters, 3, strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.Concatenate()([x, skips.pop()])

    # Output
    output = tf.keras.layers.Conv2D(num_classes, 3, padding='same', activation='softmax')(x)
    return tf.keras.Model(inputs=inputs, outputs=output)