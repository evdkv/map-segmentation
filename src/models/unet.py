'''
File containing the U-Net model architecture

Authors: Sophie Zhao, Tolya Evdokimov
'''

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout, Activation, Input
from tensorflow.keras.models import Model

class Unet:

    def __init__(self, num_classes):
        '''
        Initializes the U-Net model with the number of classes

        Parameters:
            num_classes (int): The number of classes to predict
        '''
        self.num_classes = num_classes


    def build(self):
        '''
        Builds the U-Net model

        Returns:
            Model: The U-Net model
        '''
        input = Input((128, 128, 3), name='input_1')

        en2, skip2 = self.get_encoder(input, 64)
        en3, skip3 = self.get_encoder(en2, 128)
        en4, skip4 = self.get_encoder(en3, 256)
        en5, skip5 = self.get_encoder(en4, 512)
        bottleneck = self.get_conv_tower(en5, 1024)
        de5 = self.get_decoder_conv_tower(bottleneck, skip5, 512)
        de4 = self.get_decoder_conv_tower(de5, skip4, 256)
        de3 = self.get_decoder_conv_tower(de4, skip3, 128)
        de2 = self.get_decoder_conv_tower(de3, skip2, 64)
        output = Conv2D(self.num_classes, 1, activation='softmax', padding='same', name='conv2d_18')(de2)

        return Model(inputs=input, outputs=output)

    def get_conv_tower(self, input_tensor, num_filters, padding='same', activation='relu'):
        '''
        Gets a convolutional tower with two convolutional layers and an activation layer

        Parameters:
            input_tensor (Tensor): The input tensor
            num_filters (int): The number of filters to use
            padding (str): The padding to use for the convolutional layers
            activation (str): The activation function to use
        
        Returns:
            Tensor: The output tensor
        '''
        x = Conv2D(num_filters, (3, 3), padding=padding, activation=activation)(input_tensor)
        x = Conv2D(num_filters, (3, 3), padding=padding)(x)
        x = Activation(activation)(x)

        return x
    
    def get_encoder(self, input_tensor, num_filters, padding='same', stride=2, activation='relu'):
        '''
        Gets the encoder portion of the U-Net model

        Parameters:
            input_tensor (Tensor): The input tensor
            num_filters (int): The number of filters to use
            padding (str): The padding to use for the convolutional layers
            stride (int): The stride to use for the max pooling layer
            activation (str): The activation function to use
        
        Returns:
            Tensor: The output tensor
        '''
        # Get the general conv tower for the encoder
        conv_tower = self.get_conv_tower(input_tensor, num_filters, padding, activation)
        # Get the max pooling layer and apply it to the conv tower
        # Return the conv tower for skipper connections
        x = MaxPooling2D((2, 2), stride)(conv_tower)
        x = Dropout(0.3)(x)

        return x, conv_tower
    
    def get_decoder_conv_tower(self, input_tensor, skip_tensor, num_filters, padding='same', activation='relu'):
        '''
        Gets the decoder portion of the U-Net model

        Parameters:
            input_tensor (Tensor): The input tensor
            skip_tensor (Tensor): The tensor from the encoder to concatenate with
            num_filters (int): The number of filters to use
            padding (str): The padding to use for the convolutional layers
            activation (str): The activation function to use
        
        Returns:
            Tensor: The output tensor
        '''
        x = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding=padding)(input_tensor)
        x = concatenate([x, skip_tensor])
        x = Dropout(0.3)(x)
        x = self.get_conv_tower(x, num_filters, padding, activation)

        return x