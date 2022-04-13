from cProfile import label
from utils import get_data
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
n_in = 3072
n_out = 10


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

    def _accuracy(self, Y, P):
        Y = Y.T
        P = P.T
        y_num = np.argmax(Y, axis=1)
        y_pred = np.argmax(P, axis=1)
        return np.where(y_num == y_pred, 1, 0).sum() / y_num.shape[0]

    def _shuffle(self, X, Y, Z=None):
        n = X.shape[1]
        idx = np.random.permutation(n)
        if type(Z) == type(None):
            return X[:, idx], Y[:, idx]
        return X[:, idx], Y[:, idx], Z[:, idx]

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
        self.W1, self.b1, self.W2, self.b2 = old_params

    def fit(self, X, Y, learning_rate, epochs, batch_size, x_val, y_val, lr_scheduler=None):
        n = X.shape[1]
        logs = {'train_loss': [], 'train_acc': [], 'train_cost': []}
        logs['val_loss'] = []
        logs['val_acc'] = []
        logs['val_cost'] = []
        iter = 0
        for e in range(epochs):
            X, Y = self._shuffle(X, Y)
            with tqdm(range(0, (n//batch_size)+1)) as pbar:
                for i in pbar:
                    start_index, end_index = i*batch_size, (i+1)*batch_size
                    if i*batch_size == min(n, end_index):
                        break
                    x = X[:, start_index: end_index]
                    y = Y[:, start_index: end_index]
                    p, h = self._forward(x)
                    g_W1, g_b1, g_W2, g_b2 = self._backward(x, y, p, h)
                    if type(lr_scheduler) != type(None):
                        learning_rate = lr_scheduler(iter, learning_rate)
                    self.W1 -= learning_rate * g_W1
                    self.W2 -= learning_rate * g_W2
                    self.b1 -= learning_rate * g_b1
                    self.b2 -= learning_rate * g_b2
                    loss_val, cost_val, acc_val = self.evaluate(x_val, y_val)
                    loss_train, cost_train, acc_train = self.evaluate(X, Y)
                    logs['train_loss'].append(loss_train)
                    logs['train_cost'].append(cost_train)
                    logs['train_acc'].append(acc_train)
                    logs['val_loss'].append(loss_val)
                    logs['val_cost'].append(cost_val)
                    logs['val_acc'].append(acc_val)
                    iter += 1

            print(
                f'epoch {e}/{epochs}\n\tTraining: loss:{loss_train}, cost{cost_train}, acc:{acc_train}')
            print(
                f'\tValidation: loss:{loss_val}, cost{cost_val}, acc:{acc_val}')
        return logs

    def evaluate(self, X, Y):
        P, H = self._forward(X)
        return [self._loss(Y, P), self._cost(Y, P), self._accuracy(Y, P)]


if __name__ == '__main__':
    x_train, x_val, x_test, y_train, y_val, y_test, mean, std = get_data(
        os.path.join(os.getcwd(), 'Data'))

    model = MLP(x_train.shape[0], 50, 10, 0.0)
    # p, h = model._forward(x_train)
    # model._backward(x_train, y_train, p, h)
    # model.gradient_checker(x_train[:, :30], y_train[:, :30], ComputeGradsNum)
    model.fit(x_train[:, :100], y_train[:, :100], 0.01, 200, 10, x_val, y_val)
