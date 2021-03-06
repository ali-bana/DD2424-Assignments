from utils import load_data, shuffle, flip, display_flat_image
import os
import numpy as np
from tqdm import tqdm
n_in = 3072
n_out = 10


def softmax(logits):
    logits = logits.T
    assert len(logits.shape) == 2
    s = np.max(logits, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(logits - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    probs = e_x / div
    return probs.T


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward(X, W, b, is_mbce=False):
    logits = W @ X + b
    # print(logits.max())
    if is_mbce:
        return sigmoid(logits)
    p = softmax(logits)
    return p


def accuracy(Y, P):
    Y = Y.T
    P = P.T
    y_num = np.argmax(Y, axis=1)
    y_pred = np.argmax(P, axis=1)
    return np.where(y_num == y_pred, 1, 0).sum() / y_num.shape[0]


def loss(Y, P, W, lamda):
    Y = Y.T
    P = P.T
    minus_log = -1 * np.log(np.sum(P * Y, axis=1))
    return minus_log.mean() + lamda * np.sum(np.square(W))


def mbce_loss(Y, P, W, lamda):
    l = -0.1 * ((1 - Y) * np.log(1 - P) + Y * np.log(P))
    l = l.sum(axis=0)
    return l.mean()


def compute_gradients(X, Y, W, b, P, lamda):
    n_b = X.shape[1]
    g = -(Y - P)
    dl_dw = g @ X.T / n_b + 2 * lamda * W
    dl_db = (g @ np.ones((n_b, 1))) / n_b
    return dl_dw, dl_db


def fit(X, Y, W, b, lamda, eta, batch_size, epoch, x_val=None, y_val=None, flipping=False, is_mbce=False):
    n = X.shape[1]
    logs = {'train_loss': [], 'train_acc': [], 'train_cost': []}
    if is_mbce:
        eta /= 10
    if type(x_val) != type(None):
        logs['val_loss'] = []
        logs['val_acc'] = []
        logs['val_cost'] = []
    if flipping:
        X_flipped = flip(X)
    for e in range(epoch):
        if flipping:
            X, Y, X_flipped = shuffle(X, Y, X_flipped)
        else:
            X, Y = shuffle(X, Y)
        with tqdm(range(0, (n//batch_size)+1)) as pbar:
            for i in pbar:
                if i*batch_size == min(n, (i+1)*batch_size):
                    break
                x = X[:, i*batch_size: min(n, (i+1)*batch_size)]
                if flipping:
                    x_not_fliped = X[:, i*batch_size: min(n, (i+1)*batch_size)]
                    x_flipped = X_flipped[:, i *
                                          batch_size: min(n, (i+1)*batch_size)]
                    length = min(n, (i+1)*batch_size) - i*batch_size
                    condition = np.ones((n_in, length)) * \
                        np.random.binomial(1, 0.5, length)
                    x = np.where(condition > 0.5, x_flipped, x_not_fliped)
                    # for i in range(10):
                    #     print(condition[:, i])
                    #     display_flat_image(x[:, i])
                    #     display_flat_image(x_not_fliped[:, i])
                    #     display_flat_image(x_flipped[:, i])
                    # assert False
                y = Y[:, i*batch_size: min(n, (i+1)*batch_size)]
                p = forward(x, W, b, is_mbce)
                if i % 10 == 0:
                    pbar.set_description(
                        f'{e}/{epoch}, accuracy={round(accuracy(y, p),2)}, loss={round(loss(y, p, W, lamda), 2)}')
                gW, gb = compute_gradients(x, y, W, b, p, lamda)
                W = W - eta * gW
                b = b - eta * gb
        p = forward(X, W, b, is_mbce)
        train_acc = accuracy(Y, p)
        logs['train_acc'].append(train_acc)
        train_loss = loss(Y, p, W, 0) if not is_mbce else mbce_loss(Y, p, W, 0)
        logs['train_loss'].append(train_loss)
        train_cost = loss(Y, p, W, lamda)
        logs['train_cost'].append(train_cost)
        if type(x_val) != type(None):
            p = forward(x_val, W, b, is_mbce)
            val_acc = accuracy(y_val, p)
            logs['val_acc'].append(val_acc)
            val_loss = loss(y_val, p, W, 0) if not is_mbce else mbce_loss(
                y_val, p, W, 0)
            logs['val_loss'].append(val_loss)
            val_cost = loss(y_val, p, W, lamda)
            logs['val_cost'].append(val_cost)

        reporting_str = f'epoch {e}/{epoch}, train_acc: {round(train_acc, 2)}, train_loss: {round(train_loss, 2)}, train_cost: {round(train_cost, 2)}'
        if type(x_val) != type(None):
            reporting_str += f', val_acc: {round(val_acc, 2)}, val_loss: {round(val_loss, 2)}, val_cost: {round(val_cost, 2)}'
        print(reporting_str)  # DO NOT DELETE
    return W, b, logs


def evaluate(X, Y, W, b, lamda, is_mbce=False):
    P = forward(X, W, b, is_mbce)
    return {'loss': loss(Y, P, W, 0), 'cost': loss(Y, P, W, lamda), 'acc': accuracy(Y, P)}


if __name__ == '__main__':
    x_train, x_val, x_test, y_train, y_val, y_test, mean, std = load_data(os.path.join(
        os.getcwd(), 'Data', 'cifar-10-batches-py'))

    W = np.random.normal(0, 0.1, size=(n_out, n_in))
    b = np.random.normal(0, 0.1, (n_out, 1))

    p = forward(x_train, W, b)
    fit(x_train, y_train, W, b, 1, 0.01, 100, 20, x_val, y_val, True)
