# Semantic Map Segmentation

Authors: Tolya Evdokimov, Sophie Zhao, Ginny Zhang, Win Aung

The directory contains the code for the map segmentation project.
The code conatians training and evaluation scripts for the three
models as well as implementations for the three models. Additionally,
the analyses and visualizations were done in the Jyputer notebooks.

The code for the FCN models was adopted from:
https://github.com/divamgupta/image-segmentation-keras/blob/master/keras_segmentation/models/fcn.py

The code for U-Net was compiled based on many sources from the internet
and is a generic U-Net implementation in Keras.

## Running the code
 - TFRecord data processing: Run src/data/process.py
 - Training: Change model initialization in the main() function 
 and run src/train.py
 - Testing: Change model initialization and run src/evaluate.py
 - Analyses: Run each individual notebook in notebooks/

## Project Tree (data and weights are excluded)
```bash
.
├── README.md
├── configs
│   ├── fcn32.yml
│   ├── fcn8.yml
│   └── unet.yml
├── dataset
│   ├── test.tfrecords
│   ├── train.tfrecords
│   └── valid.tfrecords
├── experiments
│   ├── fcn32
│   │   ├── model_hist_fcn32.json
│   │   └── weights
│   ├── fcn8
│   │   ├── model_hist_fcn8.json
│   │   └── weights
│   └── unet
│       ├── model_hist_unet.json
│       └── weights
├── notebooks
│   ├── dataset_analysis.ipynb
│   ├── fcn32_analysis.ipynb
│   ├── fcn8_analysis.ipynb
│   ├── unet_analysis.ipynb
│   └── weight_visualization.ipynb
└── src
    ├── data
    │   ├── process.py
    │   └── utils.py
    ├── evaluate.py
    ├── models
    │   ├── fcn32.py
    │   ├── fcn8.py
    │   └── unet.py
    └── train.py
```