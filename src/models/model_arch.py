from unet import Unet
from tensorflow.keras.utils import plot_model

model = Unet(num_classes=3).build()

plot_model(model, to_file='model.png', show_shapes=True)