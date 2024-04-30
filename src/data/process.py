import tensorflow as tf
import numpy as np
import cv2

from PIL import Image
from utils import serialize_example

COLOR_TO_CLASS = {
    (17, 141, 215): 0,
    (225, 227, 155): 1, 
    (127, 173, 123): 2,
    (185, 122, 87): 3,
    (230, 200, 181): 4,
    (150, 150, 150): 5,
    (193, 190, 175): 6
}

def seg_image_to_onehot(seg_image):

    integer_labels = np.zeros((128, 128), dtype=np.uint8)

    red_channel = seg_image[:, :, 0]  # Slice the first channel (red)
    green_channel = seg_image[:, :, 1]  # Slice the second channel (green)
    blue_channel = seg_image[:, :, 2] # Slice the third channel (blue)

    for i in range(128):
        for j in range(128):
            pixel = (red_channel[i, j], green_channel[i, j], blue_channel[i, j])
            integer_labels[i][j] = COLOR_TO_CLASS.get(pixel, 0)

    target_img = tf.convert_to_tensor(integer_labels, dtype=tf.uint8)
    target_img = tf.one_hot(target_img, 7)

    return target_img

def get_dataset():

    train_writer = tf.io.TFRecordWriter('train.tfrecords')
    test_writer = tf.io.TFRecordWriter('test.tfrecords')
    valid_writer = tf.io.TFRecordWriter('valid.tfrecords')

    base_path = "dataset/"

    train_count = 0
    valid_count = 0
    test_count = 0

    for i in range(1, 5001):

        im_base = base_path + str(i).zfill(4)
        
        raw_img = cv2.resize(np.array(Image.open(im_base + "_t.png")), (128, 128))
        seg_img = cv2.resize(np.array(Image.open(im_base + "_i2.png")), (128, 128))

        raw_img = tf.image.convert_image_dtype(raw_img, dtype=tf.float32)
        raw_img = tf.convert_to_tensor(raw_img, dtype=tf.float32)
        raw_img = tf.io.serialize_tensor(raw_img)

        one_hot_img = tf.io.serialize_tensor(seg_image_to_onehot(seg_img))

        if i < 4001:
            train_writer.write(serialize_example(raw_img, one_hot_img))
            train_count += 1
        elif i > 4000 and i < 4501:
            valid_writer.write(serialize_example(raw_img, one_hot_img))
            valid_count += 1
        elif i > 4500:
            test_writer.write(serialize_example(raw_img, one_hot_img))
            test_count += 1
    
    print(f"Summary: train: {train_count}, valid: {valid_count}, test: {test_count}")


if __name__ == '__main__':
    get_dataset()