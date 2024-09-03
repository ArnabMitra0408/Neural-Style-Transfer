import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.applications import vgg16
from IPython.display import Image, display


def preprocess_img(img_path,n_rows,n_cols):
    img=keras.utils.load_img(img_path,target_size=(n_rows,n_cols))
    img=keras.utils.img_to_array(img)
    img=np.array([img])
    img=vgg16.preprocess_input(img)
    return img