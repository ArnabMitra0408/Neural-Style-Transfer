import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.applications import vgg16
from IPython.display import Image, display


def f_extractor(layer_name,model):
    f=keras.Model(inputs=model.inputs,outputs=model.get_layer(name=layer_name).output)
    return f

def content_loss(g_f,c_f):
    return tf.reduce_sum(tf.square(g_f-c_f))


def gram_matrix(features):
    #Here I made channels the first dimension
    features = tf.transpose(features, (2, 0, 1))

    #Make the above matrix 2d. keep the channel dimension
    features = tf.reshape(features, (tf.shape(features)[0], -1))

    # Making Gram Matrix G= A* A_T
    gram_matrix = tf.matmul(features, tf.transpose(features))

    # Normalizing loss
    N = features.shape[0]
    M = features.shape[1]
    denominator = 2 * N  * M
    # return the scaled matrix
    return gram_matrix / tf.cast(denominator, tf.float32)

def style_loss(g_features, s_features):
# calculate style loss
    Gen = gram_matrix(g_features[0]) 
    
    Style = gram_matrix(s_features[0])
    return tf.reduce_sum(tf.square(Style-Gen))


def total_loss(g_img, c_img, s_img,content_weight,style_weight,model):

    # define layers for content and style
    c_layer_name = 'block5_conv2'
    s_layer_names = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1",]

    # extract feature maps for content
    feature_extractor = f_extractor(c_layer_name, model)
    c_features = feature_extractor(c_img)
    g_c_features = feature_extractor(g_img)

    # extract feature maps for style
    s_features = []
    g_s_features = []
    for s_layer in s_layer_names:
        feature_extractor = f_extractor(s_layer, model)
        s_features.append(feature_extractor(s_img))
        g_s_features.append(feature_extractor(g_img))

    # define loss variable
    loss = tf.zeros(shape=())

    # calculate content loss
    loss += content_weight * content_loss(g_c_features, c_features)

    # calculate style loss
    for i, s_feature in enumerate(s_features):
        s_loss = style_loss(g_s_features[i], s_feature)
        loss += style_weight * s_loss / len(s_features)

    return loss

def loss_gradient(g_img, c_img, s_img,content_weight,style_weight,model):
    with tf.GradientTape() as tape:
        loss = total_loss(g_img, c_img, s_img,content_weight,style_weight,model)
    grads = tape.gradient(loss, g_img)
    return loss, grads