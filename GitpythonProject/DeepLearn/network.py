"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np


class Network(object):
    def __init__(self, sizes):
        """size："""
        self.num_layers = len(sizes)  # 设置隐藏层层数
        self.sizes = sizes  # 设置神经元个数；为列表[2,3,1]：分别表示第一层，第二层，第三层的个数是2，3，1个
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]    # 从1开始，由于第一个输入不需要偏置
        self.weights = [np.random.randn(y, x)   # x, y用来生成维度(相反则是作用在转置)；randn生成一组随机正态分布数
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """输入a，得到一个新的输出a；也就是激活下一层神经元"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """使用小批量随机梯度下降法训练神经网络。traning_data是训练输入x和训练输出y组成的列表。
        如果提供test_data则会在每轮测试数据后进行预测，并输出部分进度信息。（对于追踪进度很有用，但会延长整体处理时间）
        epochs：训练轮数
        mini_batch_size：采样的小批量数据的大小
        eta：学习率
        """
        global n_test  # 获得测试数据大小
        if test_data: n_test = len(test_data)
        n = len(training_data)  # 训练数据大小
        for j in range(epochs):  # 训练次数
            random.shuffle(training_data)   # 将序列的所有元素随机排序
            mini_batches = [
                training_data[k:k + mini_batch_size]    # 将training_data进行切片处理，每个列表大小为mini_batch_size大小
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)  # 实现梯度下降
            if test_data:
                print("Epoch {0}: {1} / {2}").format(j, self.evaluate(test_data), n_test)
            else:
                print("Epoch {0} complete").format(j)

    def update_mini_batch(self, mini_batch, eta):
        """对于一个小批量应用梯度下降法和反向传播算法，用来更新w和b
        mini_batch由若干个(x, y)组成的列表， eta为学习率"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """返回测试输出中神经网络输出正确结果的数目。"""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations - y)



#### Miscellaneous functions
def sigmoid(z):
    """定义sigmoid函数"""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """sigmoid函数求导"""
    return sigmoid(z) * (1 - sigmoid(z))