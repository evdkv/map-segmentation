from tensorflow.keras.optimizers import Adam
from models.unet import Unet
import yaml
import tensorflow as tf
from json import dump
import numpy as np

from data.process import get_dataset

def main():

    config = get_config('configs/unet.yml')

    train_dataset, val_dataset, test_dataset = get_dataset(config['batch_size'])

    train_dataset = train_dataset.batch(config['batch_size'])
    val_dataset = val_dataset.batch(config['batch_size'])

    # Assuming 'data' is your data
    #data = np.squeeze(data, axis=1)

    model = Unet(num_classes=3).build()
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    output = model.predict(val_dataset)
    print("Output shape: ", output.shape)
    
    callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor=config['early_stop_monitor'], patience=config['patience']),
    tf.keras.callbacks.ModelCheckpoint(filepath=config['checkpoint_save_path'],
                                    save_weights_only=config['weights_only'],
                                    monitor=config['checkpoint_monitor'],
                                    mode=config['checkpoint_mode'],
                                    save_best_only=config['save_best'])
    ]
    model_history = model.fit(train_dataset, validation_data=val_dataset,
                        epochs=config['epochs'],
                        callbacks=callbacks,
                        verbose=config['verbose'],
                        batch_size=config['batch_size']) 
    
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


main()