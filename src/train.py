from tensorflow.keras.optimizers import Adam
from models.unet import Unet
import yaml
import tensorflow as tf
from json import dump

def main():

    config = get_config('unet.yml')

    model = Unet(num_classes=3).build()
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
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
                        verbose=config['verbose']) 
    
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
