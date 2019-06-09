import time
from progress_bar import printProgressBar
from utils import reshape
import os
import numpy as np
from multiprocessing import Process, Queue, managers
from multiprocessing.pool import ThreadPool
from multiprocessing import Manager
from functools import reduce
from operator import mul

class FeatureExtractor:

    def __call__(self, images, feature_learner, k, receptive_field_size, stride):
        self.feature_learner = feature_learner
        self.k = k
        self.receptive_field_size = receptive_field_size
        self.stride = stride

        processes = []
        n = len(images)
        N = int(os.environ['NUMBER_OF_PROCESSORS'])
        manager = Manager()
        return_dict = manager.dict()

        for i in range(N):
            p = Process(target = self.__get_images_reprezentation__, args = (images[i*n//N:(i+1)*n//N], i, return_dict))
            processes.append(p)
            p.start()

        results = []
        for i in range(N):
            processes[i].join()
            results.append(return_dict[i])

        images_reprezentation = np.concatenate(tuple(results), axis = 0)
        return images_reprezentation

    def __get_images_reprezentation__(self, images, procnum = 0, return_dict = None):
        images_reprezentation = []
        start_time = time.time()
        # progress_bar.printProgressBar(0, images.shape[0], prefix = 'Progress:', suffix = 'Complete', length = 50)
        for i in range(images.shape[0]):
            images_reprezentation.append(self.__get_classifier_features__(images[i]))
            # progress_bar.printProgressBar(i + 1, images.shape[0], prefix = 'Progress:', suffix = 'Complete', length = 50)
        images_reprezentation = np.asarray(images_reprezentation)
        print ("Timp: (s)", time.time() - start_time)
        if return_dict == None:
            return images_reprezentation
        else:
            return_dict[procnum] = images_reprezentation


    def __get_classifier_features__(self, image):
        n_row = (image.shape[0] - self.receptive_field_size) // self.stride + 1
        n_col = (image.shape[1] - self.receptive_field_size) // self.stride + 1
        image_representation = np.empty((n_row, n_col, self.k), dtype = float)
        for i in range(n_row):
            for j in range(n_col):
                receptive_field = image[i:i+self.receptive_field_size:1,j:j+self.receptive_field_size:1]
                image_representation[i][j] = self.feature_learner()(self.feature_learner, receptive_field.reshape(reduce(mul, receptive_field.shape, 1)))
        return self.__polling__(np.asarray(image_representation))

    def __polling__(self, image_repr):
        n_row = image_repr.shape[0]
        n_col = image_repr.shape[1]

        classifier_features = np.empty((4, image_repr.shape[2]), dtype = object)
        q1 = image_repr[0:n_row//2, 0:n_col//2].sum(axis = (0, 1))
        q2 = image_repr[0:n_row//2, n_col//2:n_col].sum(axis = (0, 1))
        q3 = image_repr[n_row//2:n_row, 0:n_col//2].sum(axis = (0, 1))
        q4 = image_repr[n_row//2:n_row, n_col//2:n_col].sum(axis = (0, 1))
        
        return np.append(q1, np.append(q2, np.append(q3, q4)))