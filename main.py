import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.applications import vgg16
from flask import Flask, request, render_template, send_from_directory
import os
import logging
from src.preprocess import preprocess_img
from src.loss import loss_gradient
from src.deprocess import deprocess_img
from src.utils.common_utils import read_params, sql_connect, Custom_Handler
from argparse import ArgumentParser
# Initialize Flask app
app = Flask(__name__)


# Set up logging
db = sql_connect()
custom_handler = Custom_Handler(db)
logger = logging.getLogger('NST')
logger.setLevel(logging.DEBUG)
logger.addHandler(custom_handler)

# Load parameters
configs = read_params('params.yaml')

def train(optimizer, iterations, g_img, c_img, s_img, content_weight, style_weight, model, n_rows, n_cols):
    for i in range(1, iterations + 1):
        print(f"Epoch {i}")
        logger.info(f"Epoch {i}")
        loss, grads = loss_gradient(g_img, c_img, s_img, content_weight, style_weight, model)
        print(f"Loss is {loss}")
        logger.info(f"Loss is {loss}")
        optimizer.apply_gradients([(grads, g_img)])
        if i==iterations:
            new_img = deprocess_img(np.array(g_img[0]), n_rows, n_cols)
            fname = f'static/output.jpg'
            print(f"Image generated for epoch {i} at path {fname}")
            logger.info(f"Image generated for epoch {i} at path {fname}")
            keras.utils.save_img(fname, new_img)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'content_image' not in request.files or 'style_image' not in request.files:
        return "No files uploaded", 400

    content_image = request.files['content_image']
    style_image = request.files['style_image']

    if content_image.filename == '' or style_image.filename == '':
        return "No selected file", 400

    if content_image and style_image:
        content_img_path = configs['data']['content_img_path']
        style_img_path = configs['data']['style_img_path']

        content_image.save(content_img_path)
        style_image.save(style_img_path)

        # Start style transfer
        style_weight = float(configs['weights']['style_weight'])
        content_weight = float(configs['weights']['content_weight'])
        n_rows = int(configs['weights']['n_rows'])
        w, h = keras.utils.load_img(content_img_path).size
        n_cols = int((w * n_rows) / h)
        iterations = int(configs['weights']['n_iter'])
        learning_rate = float(configs['weights']['learning_rate'])
        vgg_weights = configs['weights']['vgg_weights']
        model = vgg16.VGG16(weights=vgg_weights, include_top=False)

        g_img = tf.Variable(preprocess_img(content_img_path, n_rows, n_cols))
        c_img = preprocess_img(content_img_path, n_rows, n_cols)
        s_img = preprocess_img(style_img_path, n_rows, n_cols)

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        train(optimizer, iterations, g_img, c_img, s_img, content_weight, style_weight, model, n_rows, n_cols)

        return render_template('index.html', generated_image=True)

@app.route('/static/<path:filename>')
def serve_file(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
