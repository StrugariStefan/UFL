import numpy as np
import keras
import random
import matplotlib.pyplot as plt
from keras.datasets import cifar10

class ufl():
    def __init__(self, dataset_name, x_train, x_test, y_train, y_test, patching_probability = None, k_vector = None, selected_kernel = "k_means_soft", selected_classificator = "svc", stride = 1, receptive_field_size = 6):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.dataset_name = dataset_name
        self.patching_probability = patching_probability
        self.k_vector = k_vector
        self.receptive_field_size = receptive_field_size
        self.stride = stride
        self.selected_kernel = selected_kernel
        self.selected_classificator = selected_classificator
        self.learned_centroids = dict()
        self.images_reprezentation = dict()
        self.test_images_reprezentation = dict()
        self.STRIDE_SUFFIX = "_s" + str(self.stride) + "_r" + str(self.receptive_field_size)
        self.alfas = dict()

    def pre_training_faze(self, whitening = False, new_patches = False, dynamic_configuration = False, grayscale = False):
        print ("Pre-training faze beginning...")

        shape = self.x_train.shape[1:]
        self.x_train = self.reshape(self.x_train)
        self.x_test = self.reshape(self.x_test)
            
        self.x_train = self.reshape(self.x_train, shape = self.x_train.shape[:1] + shape)
        self.x_test = self.reshape(self.x_test, shape = self.x_test.shape[:1] + shape)

        if grayscale == True:
            self.x_train = rgb2gray(self.x_train)
            self.x_test = rgb2gray(self.x_test)

        patching_callbacks = {
            "patches": {
                "callback": self.extract_random_patches,
                "args": {
                    "images": self.x_train
                }
            },
            "stride": {
                "callback": identity_function,
                "args": {
                    "parameter": self.stride
                }
            },
            "receptive_field_size": {
                "callback": identity_function,
                "args": {
                    "parameter": self.receptive_field_size
                }
            }
        }

        data = persistance("patches", self.dataset_name, new_patches, **patching_callbacks)
        
        patches = data['patches']
        patches_shape = patches.shape[1:]
        print (patches.dtype)
        patches = self.reshape(patches)
        self.patches = patches

        
        if whitening == True and grayscale == False:
            callbacks = {
                "patches": {
                    "callback": self.whiten_images,
                    "args": {
                        "images": self.patches
                    }
                }
            }

            data = persistance("whitened", self.dataset_name, new_patches, **callbacks)
            patches = data['patches']
            self.patches = patches
        
        
        from math import log10
        from math import floor
        from math import sqrt
        from math import log2

        print (patches.shape)
        print (self.k_vector)

        print (self.x_train.shape)
        print (self.x_test.shape)

        for k in self.k_vector:
            if dynamic_configuration == True:
                alfa_callbacks = {
                    "alfa": {
                        "callback": dynamic_configure,
                        "args": {
                            "V": patches,
                            "d": 2,
                            "k": k,
                            "m": 10
                        },
                    },
                    "k": {
                        "callback": identity_function,
                        "args": {
                            "parameter": k
                        }
                    },
                    "m": {
                        "callback": identity_function,
                        "args": {
                            "parameter": 10
                        }
                    }
                }
                data = persistance("alfas", self.dataset_name, False, "_k" + str(k), **alfa_callbacks)
                alfa = data['alfa']
            else:
                alfa = 2
            self.alfas[k] = alfa

            feature_extraction_callbacks = {
                "centroids": {
                    "callback": algorithm1,
                    "args": {
                        "V": patches,
                        "d": 2,
                        "k": k,
                        "alfa": alfa,
                        "beta": 2,
                        "verbrose": True
                    }
                },
                'd': {
                    "callback": identity_function,
                    "args": {
                        "parameter": 2
                    }
                },
                "k": {
                    "callback": identity_function,
                    "args": {
                        "parameter": k
                    }
                },
                "alfa": {
                    "callback": identity_function,
                    "args": {
                        "parameter": alfa
                    }
                },
                "beta": {
                    "callback": identity_function,
                    "args": {
                        "parameter": 2
                    }
                }
            }

            data = persistance("features", self.dataset_name, False, "_k" + str(k) + "_alfa" + str(alfa) + "_beta2", **feature_extraction_callbacks)
            if len(data['centroids']) == 1:
                self.learned_centroids[k] = data['centroids']
            else:
                self.learned_centroids[k] = data['centroids'][0] 

            print ("Pre-training faze complete")

    def extract_classifier_features(self, new_image_representation = False):

        if self.learned_centroids == dict():
            self.pre_training_faze(True, False)

        print ()
        print ("Classifier feature extraction beginning")
        

        for k in self.k_vector:
            print ("\tObtaining image reprezentation with " + self.dataset_name + "_k" + str(k) + "_alfa" + str(self.alfas[k]) + "_beta2_" + self.selected_kernel + " learned features...")

            self.feature_learner = kernel[self.selected_kernel](self.learned_centroids[k])
            feature_mapper = self.feature_learner()

            images_reprezentation_callbacks = {
                "images_reprezentation": {
                    "callback": self.__get_images_reprezentation_distributed__,
                    "args": {
                        "images": self.x_train,
                        "feature_mapping": feature_mapper,
                        "k": k
                    }
                },
                "selected_kernel": {
                    "callback": identity_function,
                    "args": {
                        "parameter": self.selected_kernel
                    }
                }
            }

            print ("\t\tObtaining images reprezentations...")

            data = persistance("reprezentations", self.dataset_name, new_image_representation, "_k" + str(k) + "_alfa" + str(self.alfas[k]) + "_beta2_" + self.selected_kernel, **images_reprezentation_callbacks)
            self.images_reprezentation[k] = data['images_reprezentation']

            test_images_reprezentation_callbacks = {
                "test_images_reprezentation": {
                    "callback": self.__get_images_reprezentation_distributed__,
                    "args": {
                        "images": self.x_test,
                        "feature_mapping": feature_mapper,
                        "k": k
                    }
                },
                "selected_kernel": {
                    "callback": identity_function,
                    "args": {
                        "parameter": self.selected_kernel
                    }
                }
            }

            print ("\t\tObtaining test images reprezentations...")

            data = persistance("test_reprezentations", self.dataset_name, new_image_representation, "_k" + str(k) + "_alfa" + str(self.alfas[k]) + "_beta2_" + self.selected_kernel, **test_images_reprezentation_callbacks)
            self.test_images_reprezentation[k] = data['test_images_reprezentation']

        print ("Classifier feature extraction completed")

    


    def train_and_test(self, new_results = False):
        if bool(self.images_reprezentation) == False or bool(self.test_images_reprezentation) == False:
            self.extract_classifier_features(False)

        

        print (self.selected_classificator + " train/test beginning")
        results = []

        for k in self.k_vector:
            classificator = classification_algorithms[self.selected_classificator](name = self.dataset_name + "_k" + str(k) + "_alfa" + str(self.alfas[k]) + "_beta2_" + self.selected_kernel)
            classification_callbacks = {
                "accuracy": {
                    "callback": classificator,
                    "args": {
                        "x_train": self.images_reprezentation[k],
                        "y_train": self.y_train,
                        "x_test" : self.test_images_reprezentation[k],
                        "y_test" : self.y_test,
                        "feature_learner": kernel[self.selected_kernel](self.learned_centroids[k])
                    }
                },
                "selected_kernel": {
                    "callback": identity_function,
                    "args": {
                        "parameter": self.selected_kernel
                    }
                },
                "selected_classificator": {
                    "callback": identity_function,
                    "args": {
                        "parameter": self.selected_classificator
                    }
                },
                "k": {
                    "callback": identity_function,
                    "args": {
                        "parameter": k
                    }
                }
            }
            data = persistance("results", self.dataset_name, new_results,"_k" + str(k) + "_alfa" + str(self.alfas[k]) + "_beta2_" + self.selected_kernel + "_" + self.selected_classificator, **classification_callbacks)
            results.append(data)

        print (self.selected_classificator + " train/test completed")

        return results


if __name__ == "__main__":
    start_time = time.time()
    (xt, yt), (xte, yte) = cifar10.load_data()
    u = ufl("cifar10_50", xt[:50], xte[:10], yt[:50].flatten(), yte[:10].flatten(), 0.1, [5, 10], selected_kernel = "k_means_soft")
    u.pre_training_faze(False, False)
    u.extract_classifier_features(False)
    print (u.train_and_test(False))
    print ("Total execution time: ", time.time() - start_time)
    
