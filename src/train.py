'''
File containing the training tscript for the models
(We switch up the model to initialize inside of the main function)

Authors: Ginny Zhang, Win Aung
'''

from tensorflow.keras.optimizers import Adam
from models.unet import Unet
from json import dump

import yaml
import tensorflow as tf

from data.utils import preprocess_dataset

def main():

    # Get config
    config = get_config('configs/unet.yml')

    # Get the datasets
    raw_train = tf.data.TFRecordDataset("train.tfrecords")
    raw_valid = tf.data.TFRecordDataset("valid.tfrecords")
    raw_test = tf.data.TFRecordDataset("test.tfrecords")

    # Preprocess the datasets
    train = preprocess_dataset(raw_train, sum(1 for _ in raw_train))
    valid = preprocess_dataset(raw_valid, sum(1 for _ in raw_valid))
    test = preprocess_dataset(raw_test, sum(1 for _ in raw_test))

    # Initialize the model
    model = Unet(num_classes=7).build()
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Specify callbacks
    callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor=config['early_stop_monitor'], patience=config['patience']),
    tf.keras.callbacks.ModelCheckpoint(filepath=config['checkpoint_save_path'],
                                    save_weights_only=config['weights_only'],
                                    monitor=config['checkpoint_monitor'],
                                    mode=config['checkpoint_mode'],
                                    save_best_only=config['save_best'])
    ]

    # Train the model
    model_history = model.fit(train, validation_data=valid,
                        epochs=config['epochs'],
                        callbacks=callbacks,
                        verbose=config['verbose'],
                        batch_size=config['batch_size']) 
    
    # Save the model history
    with open(config['hist_save_path'], 'w') as hist_f:
        dump(model_history.history, hist_f)

def get_config(config_name: str) -> dict:
    '''
    Reads an experiment config file and
    returns a dictionary with the config parameters
    '''
    
    with open(config_name, 'r') as cfile:
        try:
            config = yaml.safe_load(cfile)
        except yaml.YAMLError as e:
            print(e)

    return config

if __name__ == '__main__':
    main()