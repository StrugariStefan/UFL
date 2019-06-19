from keras.datasets import mnist
from keras.datasets import cifar10
# from small_norb.smallnorb.dataset import SmallNORBDataset

def load_mnist():
    return mnist.load_data(), ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def load_cifar10():
    return cifar10.load_data(), ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

dataset_name = {
    'mnist': load_mnist,
    'cifar10': load_cifar10
}