from utils import get_data
import os
import numpy as np
from tqdm import tqdm
from functions import ComputeGradsNum
n_in = 3072
n_out = 10


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


class MLP:
    def __init__(self, n_in, n_h, n_out, lambda_) -> None:
        self.W1 = np.random.normal(0, 1/np.sqrt(n_h), (n_h, n_in))
        self.b1 = np.zeros(n_h)
        self.W2 = np.random.normal(0, 1/np.sqrt(n_out), (n_out, n_h))
        self.b2 = np.zeros(n_out)
        self.lambda_ = lambda_

    def _softmax(self, logits):
        logits = logits.T
        assert len(logits.shape) == 2
        s = np.max(logits, axis=1)
        s = s[:, np.newaxis]
        e_x = np.exp(logits - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis]
        probs = e_x / div
        return probs.T

    def _relu(self, logits):
        return np.where(logits > 0, logits, 0)

    def _forward(self, X):
        h = self._relu(((self.W1 @ X).T + self.b1).T)
        p = self._softmax(((self.W2 @ h).T + self.b2).T)
        return p, h

    def _backward(self, X, Y, P, H):
        n_batch = X.shape[1]
        gradient = -(Y - P)
        g_W2 = gradient @ H.T / n_batch + 2 * self.lambda_ * self.W2
        g_b2 = np.mean(gradient, axis=1)
        gradient = self.W2.T @ gradient
        gradient = gradient * np.where(H > 0, 1, 0)
        g_W1 = gradient @ X.T / n_batch + 2 * self.lambda_ * self.W1
        g_b1 = np.mean(gradient, axis=1)
        return g_W1, g_b1, g_W2, g_b2

    def _loss(self, Y, P):
        minus_log = -1 * np.log(np.sum(P * Y, axis=0))
        return minus_log.mean()

    def _cost(self, Y, P):
        return self._loss(Y, P) + self.lambda_ * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))

    def _custom_cost(self, X, Y, W1, b1, W2, b2, lambda_):
        h = self._relu(((W1 @ X).T + b1).T)
        p = self._softmax(((W2 @ h).T + b2).T)
        return self._loss(Y, p) + lambda_ * (np.sum(np.square(W1)) + np.sum(np.square(W2)))

    def gradient_checker(self, X, Y, checker_function):
        # changes the weights and bias for a bigger gradeint
        old_params = [self.W1, self.b1, self.W2, self.b2]
        self.W1 = np.random.uniform(-1, 1, self.W1.shape)
        self.W2 = np.random.uniform(-1, 1, self.W2.shape)
        self.b1 = np.random.uniform(-1, 1, self.b1.shape)
        self.b2 = np.random.uniform(-1, 1, self.b2.shape)
        P, H = self._forward(X)
        g_w1, g_b1, g_w2, g_b2 = self._backward(X, Y, P, H)
        g_w1_num, g_b1_num, g_w2_num, g_b2_num = checker_function(
            X, Y, self.W1, self.b1, self.W2, self.b2, self.lambda_, self._custom_cost)
        dif_w1 = np.abs(g_w1 - g_w1_num)
        dif_w2 = np.abs(g_w2 - g_w2_num)
        dif_b1 = np.abs(g_b1 - g_b1_num)
        dif_b2 = np.abs(g_b2 - g_b2_num)
        print(f'for W1: Max:{np.abs(g_w1_num).max()}, mean:{g_w1_num.mean()}, std:{g_w1_num.std()} \nfor W1 error: Max:{dif_w1.max()}, mean:{dif_w1.mean()}, std:{dif_w1.std()}')

        print(f'for W2: Max:{np.abs(g_w2_num).max()}, mean:{g_w2_num.mean()}, std:{g_w2_num.std()} \nfor W2 error: Max:{dif_w2.max()}, mean:{dif_w2.mean()}, std:{dif_w2.std()}')

        print(f'for b1: Max:{np.abs(g_b1_num).max()}, mean:{g_b1_num.mean()}, std:{g_b1_num.std()} \nfor b1 error: Max:{dif_b1.max()}, mean:{dif_b1.mean()}, std:{dif_b1.std()}')

        print(f'for b2: Max:{np.abs(g_b2_num).max()}, mean:{g_b2_num.mean()}, std:{g_b2_num.std()} \nfor b2 error: Max:{dif_b2.max()}, mean:{dif_b2.mean()}, std:{dif_b2.std()}')


if __name__ == '__main__':
    x_train, x_val, x_test, y_train, y_val, y_test, mean, std = get_data(
        os.path.join(os.getcwd(), 'Data'))

    model = MLP(x_train.shape[0], 50, 10, 0.1)
    # p, h = model._forward(x_train)
    # model._backward(x_train, y_train, p, h)
    model.gradient_checker(x_train[:, :30], y_train[:, :30], ComputeGradsNum)
