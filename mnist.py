if __name__ == '__main__':
    import keras
    from keras.datasets import mnist
    import numpy as np
    from utils import Persistance
    import csv
    from imageio import imwrite
    from math import log10, ceil
    import random
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print (x_train.shape)
    print (x_test.shape)
    print (y_train.shape)
    print (y_test.shape)


    r1 = 2000
    n1 = len(x_train)
    x1 = {i for i in range(n1)}
    xs1 = set(random.sample(x1, r1))

    r2 = 400
    n2 = len(x_test)
    x2 = {i for i in range(n2)}
    xs2 = set(random.sample(x2, r2))

    print (xs1)
    print (xs2)


    x_train_raw = x_train[np.array(list(xs1))]
    x_test_raw = x_test[np.array(list(xs2))]
    y_train = y_train[np.array(list(xs1))].flatten()
    y_test = y_test[np.array(list(xs2))].flatten()

    print (x_train_raw.shape)
    print (x_test_raw.shape)

    data = dict()
    data['x_train_raw'] = x_train_raw
    data['x_test_raw'] = x_test_raw
    data['y_train'] = y_train
    data['y_test'] = y_test
    data['labels'] = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    p = Persistance("datasets")
    p.save(data, "mnist_2000", "")

    data, arguments = p.load("mnist_2000", "")
    print (data, arguments)

    xt_len = ceil(log10(len(x_train_raw)))
    for i, image in enumerate(x_train_raw):
        istr = str(i)
        imwrite('mnist_example2000\\train\\' + istr.zfill(xt_len) + '.jpg', image)

    xte_len = ceil(log10(len(x_test_raw)))
    for i, image in enumerate(x_test_raw):
        istr = str(i)
        imwrite('mnist_example2000\\test\\' + istr.zfill(xte_len) + '.jpg', image)

    train_labels = []
    for i, label in enumerate(y_train):
        train_labels.append(label)

    with open('mnist_example2000\\train.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(train_labels)

    test_labels = []
    for i, label in enumerate(y_test):
        test_labels.append(label)

    with open('mnist_example2000\\test.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(test_labels)

    with open('mnist_example2000\\labels.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(data['labels'])