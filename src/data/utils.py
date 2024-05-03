'''
File containing utility functions for the data pipeline to preprocess the dataset
and generate the .tfrecords files

Authors: Ginny Zhang, Win Aung, Tolya Evdokimov, Sophie Zhao
'''

import tensorflow as tf

FEATURE_DESCRIPTION = {
    'raw_image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.string)
}

AUTOTUNE = tf.data.AUTOTUNE

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(raw_img, seg_img):
    '''Serialize features of a single example and return a serialized string'''
    feature = {
        'raw_image' : _bytes_feature(raw_img),
        'label' : _bytes_feature(seg_img)
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def _decode_example(example):
    '''Decodes the X and Y tesnors for training'''
    example['raw_image'] = tf.ensure_shape(tf.io.parse_tensor(example['raw_image'], out_type=tf.float32), (128, 128, 3))
    example['label'] = tf.ensure_shape(tf.io.parse_tensor(example['label'], out_type=tf.float32), (128, 128, 7))
    return example

def _parse_dataset(example_proto):
    '''Parses a single example from the .tfrecords file using a feature description'''
    return tf.io.parse_single_example(example_proto, FEATURE_DESCRIPTION)

def _prepare_ds(e):
    '''Separates a signle example into features and labels for model training'''
    features = {'input_1' : e['raw_image']}
    label = {'conv2d_18' : e['label']}
    return features, label

def preprocess_dataset(ds, ds_size):
    '''Combines dataset decoding, processing, and batching'''
    parsed = ds.map(_parse_dataset)
    decoded = parsed.map(_decode_example)
    prep = decoded.map(_prepare_ds)

    # Shuffle and batch the dataset (uncomment the shuffle for training)
    return prep.batch(128, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)#shuffle(buffer_size=ds_size, seed=19384)