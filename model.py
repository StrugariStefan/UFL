from feature_extraction import FeatureExtractor
import numpy as np
from functools import reduce
from operator import mul
import time

class Model:
    def __init__(self, classifier, feature_learner, train_reprezentation, test_reprezentation, x_train, y_train, x_test, y_test, k, receptive_field_size, stride, labels):
        self.classifier = classifier
        self.feature_learner = feature_learner
        self.train_reprezentation = train_reprezentation
        self.test_reprezentation = test_reprezentation
        self.x_train = x_train
        self.y_train = y_train
        self.y_test = y_test
        self.x_test = x_test
        self.receptive_field_size = receptive_field_size
        self.stride = stride
        self.k = k
        self.labels = labels

    def predict(self, images):
        img_repr = self.__get_images_reprezentation__(images)
        pred = []
        for i in range(len(img_repr)):
            pred.append(self.classifier.predict(img_repr[i:i+1]))
        return pred

    def __get_images_reprezentation__(self, images):
        images_reprezentation = []
        for i in range(images.shape[0]):
            images_reprezentation.append(self.__get_classifier_features__(images[i]))
        images_reprezentation = np.asarray(images_reprezentation)
        return images_reprezentation

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