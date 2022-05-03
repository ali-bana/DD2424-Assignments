import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import Model
import os
from utils import get_all_data, softmax, relu

x_train, x_val, x_test, y_train, y_val, y_test, mean, std = get_all_data(
    os.path.join(os.pardir, 'Data', 'cifar-10-batches-py'))

model = Model(x_train.shape[0], [10], 10, 0.01)

Ws = []
bs = []
for l in model.layers:
    Ws.append(l.W)
    bs.append(l.b)


def compute_cost(X, Y, Ws, bs, lambda_):
    t = X.copy()
    for i in range(len(Ws)-1):
        t_old = t
        t = relu(((Ws[i] @ t).T + bs[i]).T)
    t = softmax(((Ws[-1] @ t).T + bs[-1]).T)
    return


compute_cost(x_val[:, :2], y_val[:, :2], Ws, bs, 0.01)
