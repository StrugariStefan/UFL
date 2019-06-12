if __name__ == '__main__':
    import keras
    from keras.datasets import mnist
    import numpy as np
    from utils import Persistance
    import csv
    from imageio import imwrite
    from math import log10, ceil
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
    data['labels'] = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    p = Persistance("datasets")
    p.save(data, "mnist_200", "")

    data, arguments = p.load("mnist_200", "")
    print (data, arguments)

    xt_len = ceil(log10(len(x_train_raw)))
    for i, image in enumerate(x_train_raw):
        istr = str(i)
        imwrite('mnist_example\\train\\' + istr.zfill(xt_len) + '.jpg', image)

    xte_len = ceil(log10(len(x_test_raw)))
    for i, image in enumerate(x_test_raw):
        istr = str(i)
        imwrite('mnist_example\\test\\' + istr.zfill(xte_len) + '.jpg', image)

    train_labels = []
    for i, label in enumerate(y_train):
        train_labels.append(label)

    with open('mnist_example\\train.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(train_labels)

    test_labels = []
    for i, label in enumerate(y_test):
        test_labels.append(label)

    with open('mnist_example\\test.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(test_labels)

    with open('mnist_example\\labels.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(data['labels'])