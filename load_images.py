import os
from skimage.io import imread
import numpy as np
from skimage.transform import rescale, resize, downscale_local_mean


def load_images(dirpath, width, height, to_resize = False):
    all_images = []
    filenames = []

    print (width, height)

    if not os.path.isdir(dirpath):
        raise NotADirectoryError

    for image_path in os.listdir(dirpath):
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = imread(os.path.join(dirpath, image_path) , as_gray=False)
            # print (img)
            if to_resize:
                img = resize(img, (width, height), preserve_range = True,
                                anti_aliasing=False)
            all_images.append(np.copy(img).astype('uint8'))
            filenames.append(image_path)

    return np.asarray(all_images), filenames 

def load_labels(filepath, type = int):
    import csv

    labels = []
    with open(filepath, 'r', newline='') as myfile:
        r = csv.reader(myfile)
        for i, line in enumerate(r):
            labels.extend([type(el) for el in line])

    return labels