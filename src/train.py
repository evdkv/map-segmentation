from tensorflow.keras.optimizers import Adam
from models.unet import Unet
import yaml
import tensorflow as tf
from json import dump
import numpy as np

from data.utils import preprocess_dataset

def main():

    config = get_config('configs/unet.yml')

    raw_train = tf.data.TFRecordDataset("train.tfrecords")
    raw_valid = tf.data.TFRecordDataset("valid.tfrecords")
    raw_test = tf.data.TFRecordDataset("test.tfrecords")

    train = preprocess_dataset(raw_train, sum(1 for _ in raw_train))
    valid = preprocess_dataset(raw_valid, sum(1 for _ in raw_valid))
    test = preprocess_dataset(raw_test, sum(1 for _ in raw_test))

    model = Unet(num_classes=7).build()
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor=config['early_stop_monitor'], patience=config['patience']),
    tf.keras.callbacks.ModelCheckpoint(filepath=config['checkpoint_save_path'],
                                    save_weights_only=config['weights_only'],
                                    monitor=config['checkpoint_monitor'],
                                    mode=config['checkpoint_mode'],
                                    save_best_only=config['save_best'])
    ]
    model_history = model.fit(train, validation_data=valid,
                        epochs=config['epochs'],
                        callbacks=callbacks,
                        verbose=config['verbose'],
                        batch_size=config['batch_size']) 
    
    with open(config['hist_save_path'], 'w') as hist_f:
        dump(model_history.history, hist_f)
        
    print("Evaluate on test data")
    results = model.evaluate(test, batch_size=128)
    print("test loss, test acc:", results)

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


main()