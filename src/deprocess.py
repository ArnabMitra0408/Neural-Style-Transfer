import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.applications import vgg16
from IPython.display import Image, display

def deprocess_img(x,n_rows,n_cols):
# input - BGR image with means subtracted
    x = x.reshape((n_rows, n_cols, 3))
    # mean BGR values from imagenet - subtracted in vgg preprocess
    mean_imagenet = [103.939, 116.779, 123.68]
    # re-add means
    for i, val in enumerate(mean_imagenet):
        x[:, :, i] += val
    # restore RGB from BGR using slicing array
    x = x[:, :, ::-1] # channels put in reverse order
    return x