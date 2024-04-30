import tensorflow as tf
from models.unet import Unet
from tensorflow.keras.optimizers import Adam
from data.utils import preprocess_dataset


def main():

    raw_test = tf.data.TFRecordDataset("test.tfrecords")
    test = preprocess_dataset(raw_test, sum(1 for _ in raw_test))

    model = Unet(num_classes=7).build()
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    model.load_weights("experiments/unet/weights/unet_49-0.14.h5")

    results = model.evaluate(test, batch_size=128)
    print("test loss, test acc:", results)


if __name__ == '__main__':
    main()