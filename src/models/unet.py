from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, Input
from tensorflow.keras.models import Model

class Unet:

    def __init__(self, num_classes):
        self.num_classes = num_classes


    def build(self):
        input = Input((512, 512, 3))

        en1, skip1 = self.get_encoder(input, 32)
        en2, skip2 = self.get_encoder(en1, 64)
        en3, skip3 = self.get_encoder(en2, 128)
        en4, skip4 = self.get_encoder(en3, 256)
        en5, skip5 = self.get_encoder(en4, 512)
        bottleneck = self.get_conv_tower(en5, 1024)
        de5 = self.get_decoder_conv_tower(bottleneck, skip5, 512)
        de4 = self.get_decoder_conv_tower(de5, skip4, 256)
        de3 = self.get_decoder_conv_tower(de4, skip3, 128)
        de2 = self.get_decoder_conv_tower(de3, skip2, 64)
        de1 = self.get_decoder_conv_tower(de2, skip1, 32)
        output = Conv2D(self.num_classes, 1, activation='softmax')(de1)

        return Model(inputs=input, outputs=output)

    def get_conv_tower(self, input_tensor, num_filters, padding='same', activation='relu'):
        x = Conv2D(num_filters, (3, 3), padding=padding, activation=activation)(input_tensor)
        x = Conv2D(num_filters, (3, 3), padding=padding)(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation(activation)(x)

        return x
    
    def get_encoder(self, input_tensor, num_filters, padding='same', stride=2, activation='relu'):
        # Get the general conv tower for the encoder
        conv_tower = self.get_conv_tower(input_tensor, num_filters, padding, activation)
        # Get the max pooling layer and apply it to the conv tower
        # Return the conv tower for skipper connections
        x = MaxPooling2D((2, 2), stride)(conv_tower)

        return x, conv_tower
    
    def get_decoder_conv_tower(self, input_tensor, skip_tensor, num_filters, padding='same', activation='relu'):
        x = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding=padding)(input_tensor)
        x = concatenate([x, skip_tensor])
        x = self.get_conv_tower(x, num_filters, padding, activation)

        return x

    