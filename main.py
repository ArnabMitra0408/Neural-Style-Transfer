import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.applications import vgg16
from IPython.display import Image, display
from src.preprocess import preprocess_img
from src.loss import loss_gradient
from src.deprocess import deprocess_img
from src.utils.common_utils import read_params
from argparse import ArgumentParser
import logging
from src.utils.common_utils import sql_connect,Custom_Handler
db = sql_connect()
custom_handler = Custom_Handler(db)
logger = logging.getLogger('nst')
logger.setLevel(logging.DEBUG)
logger.addHandler(custom_handler)

def train(optimizer,iterations,g_img, c_img, s_img,content_weight,style_weight,model,n_rows,n_cols):

    for i in range(1, iterations+1):
        print(f"Epoch {i}")
        logger.info(f"Epoch {i}")
        loss, grads = loss_gradient(g_img, c_img, s_img,content_weight,style_weight,model)
        print(i, loss)
        logger.info(f"Loss is {loss}")
        optimizer.apply_gradients([(grads, g_img)])
        if i % 1000 == 0:
            new_img = deprocess_img(np.array(g_img[0]),n_rows,n_cols)
            fname = f"{g_img_path}/output_{i}.png"
            print(f"Image generated for epoch {i} at path {fname}")
            logger.info(f"Image generated for epoch {i} at path {fname}")
            keras.utils.save_img(fname, new_img)
    return

def check_gpu_avail():
    print(f"Using Tensorflow Version {tf.__version__}")
    logging.info(f"Using Tensorflow Version {tf.__version__}")
    gpu_available = tf.test.is_built_with_cuda()

    gpus = tf.config.list_physical_devices('GPU')

    if gpu_available and gpus:
        print("TensorFlow is running on the GPU.")
        logger.info("TensorFlow is running on the GPU.")
        print("Available GPU(s):", gpus)
        logger.info(f"Available GPU(s):{gpus}")
    else:
        print("TensorFlow is not running on the GPU.")
        logger.critical("TensorFlow is not running on the GPU.")
    return

if __name__=='__main__':
    args=ArgumentParser()
    args.add_argument("--config_path", '-c', default='params.yaml')
    parsed_args=args.parse_args()
    configs=read_params(parsed_args.config_path)


    check_gpu_avail()
    

    c_img_path=configs['data']['content_dir']
    c_img_path=c_img_path+'/nyc.jpg'
    s_img_path=configs['data']['style_dir']
    s_img_path=s_img_path+'/van_gogh.jpg'
    g_img_path=configs['data']['gen_dir']

    logger.info(f"Content Image Path: {c_img_path}")
    logger.info(f"Style Image Path: {s_img_path}")

    
    style_weight = 1e-3
    content_weight = 3.5e-8
    

    n_rows= int(configs['weights']['n_rows'])
    w,h=keras.utils.load_img(c_img_path).size
    n_cols = int((w * n_rows)/h)

    iterations=int(configs['weights']['n_iter'])
    learning_rate=float(configs['weights']['learning_rate'])
    vgg_weights=configs['weights']['vgg_weights']
    model = vgg16.VGG16(weights=vgg_weights, include_top=False)

    g_img = tf.Variable(preprocess_img(c_img_path,n_rows,n_cols))
    c_img = preprocess_img(c_img_path,n_rows,n_cols)
    s_img = preprocess_img(s_img_path,n_rows,n_cols)

    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate)
    
    train(optimizer,iterations,g_img, c_img, s_img,content_weight,style_weight,model,n_rows,n_cols)