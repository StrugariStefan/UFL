if __name__ == '__main__':
    import keras
    from keras.datasets import mnist
    import numpy as np
    from utils import Persistance

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print (x_train.shape)
    print (x_test.shape)
    print (y_train.shape)
    print (y_test.shape)


    x_train_raw = x_train[:200]
    x_test_raw = x_test[:40]
    y_train = y_train[:200].flatten()
    y_test = y_test[:40].flatten()

    data = dict()
    data['x_train_raw'] = x_train_raw
    data['x_test_raw'] = x_test_raw
    data['y_train'] = y_train
    data['y_test'] = y_test

    p = Persistance("datasets")
    p.save(data, "mnist_200", "")

    data, arguments = p.load("mnist_200", "")
    print (data, arguments)
