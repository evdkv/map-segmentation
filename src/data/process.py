import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from PIL import Image
from keras.utils import load_img
from PIL import ImageOps
import matplotlib.pyplot as plt

import keras
import numpy as np
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io

def load_img_masks(input_path, target_path):
    input_img = tf_io.read_file(input_path)
    input_img = tf_io.decode_png(input_img, channels=3)
    input_img = tf_image.convert_image_dtype(input_img, "float32")

    target_img = tf_io.read_file(target_path)
    target_img = tf_io.decode_png(target_img, channels=3)
    target_img = tf_image.convert_image_dtype(target_img, "uint8")

    return input_img, target_img

def get_dataset(batch_size):
    path_init = "../../dataset/testdata"
    img_size = (512,512)

    raw_paths = []
    seg_paths = []

    for fname in os.listdir(path_init):
        if fname.endswith("t.png"):
            raw_paths.append(os.path.join(path_init, fname))
    raw_paths.sort()

    for fname in os.listdir(path_init):
        if fname.endswith("i2.png"):
            seg_paths.append(os.path.join(path_init, fname))
    seg_paths.sort()

    dataset = tf_data.Dataset.from_tensor_slices((raw_paths, seg_paths))
    dataset = dataset.map(load_img_masks, num_parallel_calls=tf_data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset


# Display pairs of vectorized images
# for images, masks in dataset.take(1):
#     image = images[4]
#     mask = masks[4]
#     break
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# print(image.numpy().shape)
# ax[0].imshow(image.numpy())
# ax[0].set_title('Input Image')
# ax[0].axis('off')  # Turn off axis labels
# ax[1].imshow(mask.numpy(), cmap='gray')  # Use grayscale color map for the mask
# ax[1].set_title('Mask Image')
# ax[1].axis('off')  # Turn off axis labels
# plt.show()
