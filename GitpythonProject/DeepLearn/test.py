import random
import numpy as np

#
# sizes = np.asarray()
# x = [np.random.randn(y, 1) for y in sizes[1:]]
# sizes = [2,3,1]
# print(sizes[:-1])
# for x, y in zip(sizes[:-1], sizes[1:]):
#     print(x, y)
# x = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
# print(x)
# print([np.random.randn(y, 1) for y in sizes[1:]])

training_data = [12,33,44,55,213,52]
mini_batch_size = 3
n = len(training_data)
mini_batches = [
    training_data[k:k + mini_batch_size]
    for k in range(0, n, mini_batch_size)]
print(mini_batches)
for mini_batch in mini_batches:
    print(mini_batch)