import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
import tensorflow as tf
from tensorflow import data as tf_data
# from tensorflow import image as tf_image
# from tensorflow import io as tf_io

import numpy as np
from PIL import Image

COLOR_TO_CLASS = {
    (17, 141, 215): 0,
    (225, 227, 155): 1, 
    (127, 173, 123): 2,
    (185, 122, 87): 3,
    (230, 200, 181): 4,
    (150, 150, 150): 5,
    (193, 190, 175): 6
}

def load_img_masks(x_img, y_img):
    x_img = tf.image.convert_image_dtype(x_img, dtype=tf.float32)
    x_img = tf.convert_to_tensor(x_img, dtype=tf.float32)
    x_img = x_img / 255.0 # Normalize the images to [0, 1]
    #x_img = tf_io.decode_png(x_img, channels=3)
    #x_img = tf_image.convert_image_dtype(x_img, "float32")
    
    return x_img, y_img


def seg_image_to_onehot(seg_image):

    integer_labels = np.zeros((512, 512), dtype=np.uint8)

    #target_img = tf_io.read_file(target_path)
    #target_img = tf_io.decode_png(target_img, channels=3)
    #target_img = tf_image.convert_image_dtype(target_img, "uint8")

    red_channel = seg_image[:, :, 0]  # Slice the first channel (red)
    green_channel = seg_image[:, :, 1]  # Slice the second channel (green)
    blue_channel = seg_image[:, :, 2] # Slice the third channel (blue)

    for i in range(512):
        for j in range(512):
            pixel = (red_channel[i, j], green_channel[i, j], blue_channel[i, j])
            integer_labels[i][j] = COLOR_TO_CLASS.get(pixel, 0)

    target_img = tf.convert_to_tensor(integer_labels, dtype=tf.uint8)
    target_img = tf.one_hot(target_img, 7)

    return target_img

def get_dataset(batch_size):
    path_init = "dataset"

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

    print("reading raw images")

    raw_images = [np.array(Image.open(path)) for path in raw_paths]
    seg_images = [np.array(Image.open(path)) for path in seg_paths]


    print("converting seg images to onehot")

    for i in range(len(seg_images)):
        seg_images[i] = seg_image_to_onehot(seg_images[i])

    print("creating dataset")

    dataset = tf_data.Dataset.from_tensor_slices((raw_images, seg_images))
    dataset = dataset.map(load_img_masks, num_parallel_calls=tf_data.AUTOTUNE)
    # dataset = dataset.map(_remove_unnecessary_dimension, num_parallel_calls=tf_data.AUTOTUNE)

    #total_size = len(list(dataset.as_numpy_iterator()))
    #train_size = int(0.9 * total_size)

    train_dataset, val_test_dataset = keras.utils.split_dataset(dataset, left_size = 0.8, shuffle = True)
    val_dataset, test_dataset = keras.utils.split_dataset(val_test_dataset, left_size = 0.5, shuffle = True)

    train_dataset = train_dataset
    val_dataset = val_dataset
    test_dataset = test_dataset

    # for images, masks in train_dataset.take(1):
    #     image = images[0]
    #     mask = masks[0]

    #     print(image.shape)
    #     print(mask.shape)

    #     print(mask)
    #     break

    return train_dataset, val_dataset, test_dataset

#get_dataset(5)
# def _remove_unnecessary_dimension(image, label):
#     print(image.shape)
#     image = tf.reshape(image, [image.shape[0], image.shape[2], image.shape[3], image.shape[4]])
#     label = tf.reshape(label, [label.shape[0], label.shape[2], label.shape[3], label.shape[4]])
#     return image, label

# train_dataset, val_dataset, test_dataset = get_dataset(5)

# Display pairs of vectorized images

# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# print(image.numpy().shape)
# ax[0].imshow(image.numpy())
# ax[0].set_title('Input Image')
# ax[0].axis('off')  # Turn off axis labels
# ax[1].imshow(mask.numpy(), cmap='gray')  # Use grayscale color map for the mask
# ax[1].set_title('Mask Image')
# ax[1].axis('off')  # Turn off axis labels
# plt.show()
