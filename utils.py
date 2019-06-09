import numpy as np
import matplotlib.pyplot as plt
from operator import mul
from functools import reduce
from pathlib import Path
import os

def reshape(images, shape = None):
    if shape == None:
        return images.reshape(images.shape[0], reduce(mul, images.shape[1:], 1))
    return images.reshape(shape)

def plotImage(image, shape):
    plt.figure(figsize=(1.5, 1.5))
    plt.imshow(image.reshape(shape[0], shape[1], shape[2]))
    plt.show()
    plt.close()

def identity_function(parameter):
    return parameter

class Persistance:

    def __init__(self, dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.dir_path = dir_path

    def save(self, data, prefix, suffix, **args):
        base_name = self.dir_path + "\\" + prefix + suffix 
        np.savez_compressed(base_name, data = data, arguments = args)

    def load(self, prefix, suffix):
        base_name = self.dir_path + "\\" + prefix + suffix 
        if Path(base_name + ".npz").is_file():
            f = np.load(base_name + ".npz")
            try:
                data = f['data'].item()
            except ValueError:
                data = f['data']
            arguments = f['arguments'].item()
            return data, arguments
        else:
            raise FileNotFoundError()