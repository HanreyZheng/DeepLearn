import gzip
import _pickle
import numpy as np
# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# print(training_data)


def load_data():
    """以元组的形式返回MNIST数据，包含训练数据、验证数据和测试数据
    training_data->(x, y) x->实际的训练图像，5w项的Numpy ndarry。每项有着784个值的Numpy ndarry，代表28*28=784的像素；
    y->包含数字0~9
    validation_data和test_data类似，但是只有1w幅"""
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = _pickle.load(f, encoding='bytes')
    f.close()
    return (training_data, validation_data, test_data)