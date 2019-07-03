import numpy as np
import time
from progress_bar import printProgressBar
from utils import identity_function, reshape
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

def rgb2gray(rgb_images):
    return np.dot(rbg_images[...,:3], [0.2989, 0.5870, 0.1140])

def normalize(images):
    images_norm = images / 255.
    images_norm -= images_norm.mean(axis = 0)
    
    images_norm = np.float32(images_norm)
    return images_norm

def tf_whitening(images):
    sess = tf.Session()
    img = None
    with sess.as_default():
        img = tf.map_fn(lambda image: tf.image.per_image_standardization(image) if len(image.shape) == 3 else image, images)
        img = img.eval()
    return img

def whiten_images(images):
    shape = images.shape[1:]
    images = reshape(images)

    n = len(images) // 5000

    if n <= 1:
        print ("Whitening dataset...")
        start_time = time.time()

        cov = np.cov(images, rowvar = True)
        U, S, V = np.linalg.svd(cov)
        
        epsilon = 0.1
        X_ZCA = U.dot(np.diag(1.0/np.sqrt(S + epsilon))).dot(U.T).dot(images)
        X_ZCA_rescaled = (X_ZCA - X_ZCA.min()) / (X_ZCA.max() - X_ZCA.min())
        print ("Time: (s)", round(time.time() - start_time, 2))
        return reshape(X_ZCA_rescaled, X_ZCA_rescaled.shape[:1] + shape)
    else:
        X_ZCA_rescaled = np.empty(tuple([0] + list(images.shape[1:])))
        for i in range(n - 1):
            print (X_ZCA_rescaled.shape)
            X_ZCA_rescaled = np.concatenate((X_ZCA_rescaled, whiten_images(images[i*5000:(i+1)*5000])), axis = 0)
            
        return X_ZCA_rescaled

def extract_random_patches(images, nextf, receptive_field_size = 6, stride = 1, patching_probability = 0.01):
    n_row = (images.shape[1] - receptive_field_size) // stride + 1
    n_col = (images.shape[2] - receptive_field_size) // stride + 1   
    patches = []

    if patching_probability == None:
        patching_probability = 20000 / (n_row * n_col * images.shape[0])

    print ("Extracting random patches...")
    start_time = time.time()
    count = 0
    for image_index in range(images.shape[0]):
        for i in range(n_row):
            for j in range(n_col):
                z = random.uniform(0, 1)
                if z < patching_probability:
                    count += 1
                    patches.append(images[image_index][i:i+receptive_field_size:1,j:j+receptive_field_size:1])

    patches = np.asarray(patches)
    
    print ("Time: (s)", round(time.time() - start_time, 2))
    patches = np.float32(patches / 255.)
    for func in nextf:
        patches = func(patches)
    return patches


preprocessing_algorithms = {
    "whitening": whiten_images,
    "nothing": identity_function,
    "tf_whitening": tf_whitening 
}