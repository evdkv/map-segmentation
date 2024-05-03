'''
File to evaluate the model on the test dataset.
(We switch up the model to initialize inside of the main function)

Authors: Ginny Zhang, Win Aung
'''

import tensorflow as tf
from models.unet import Unet
from models.fcn8 import fcn_8
from models.fcn32 import fcn_32
from tensorflow.keras.optimizers import Adam
from data.utils import preprocess_dataset


def main():

    raw_test = tf.data.TFRecordDataset("dataset/test.tfrecords")
    test = preprocess_dataset(raw_test, sum(1 for _ in raw_test))

    model = Unet(num_classes=7).build()
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights("experiments/unet/weights/unet_49-0.14.h5")

    results = model.evaluate(test, batch_size=128)
    print("test loss, test acc:", results)

if __name__ == '__main__':
    main()

    # FCN-8: test loss, test acc: [1.8420100212097168, 0.49652379751205444]
    # FCN-32: test loss, test acc: [1.708881139755249, 0.3786005973815918]
    # U-Net: test loss, test acc: [0.13887286186218262, 0.9511458873748779]
