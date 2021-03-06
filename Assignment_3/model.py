from calendar import leapdays
import enum
from math import gamma
from turtle import forward

from matplotlib.ft2font import LOAD_LINEAR_DESIGN
from utils import get_all_data, softmax, relu
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# n_in = 3072
# n_out = 10


class Dense_layer:
    def __init__(self, n_input, n_units, activation, lambda_, use_batch_norm, after=False, adam=False) -> None:
        self.n_input = n_input
        self.n_units = n_units
        assert activation in ['relu', 'softmax']
        self.activation = activation
        self.W = np.random.normal(0, 1/np.sqrt(n_input), (n_units, n_input))
        self.b = np.zeros(n_units)
        self.gradient = (None, None)
        self.lambda_ = lambda_
        self.use_bn = use_batch_norm
        if self.use_bn:
            self.mu = np.zeros(n_units)
            self.mu_average = np.zeros(n_units)
            self.v = np.ones(n_units)
            self.v_average = np.ones(n_units)
            self.gamma = np.ones(n_units)
            self.beta = np.zeros(n_units)
            self.alpha = 0.9
        self.after = after
        self.adam_m = [0,  0, 0, 0]
        self.adam_v = [0, 0, 0, 0]
        self.t = 1
        self.adam = adam

    def forward(self, X, test, set_average=False):
        self.input = X
        h = ((self.W @ X).T + self.b).T
        if self.after and self.use_bn:
            h = relu(h)
        if self.use_bn:
            if test:
                mean = self.mu_average
                var = self.v_average
            else:
                self.s = h.copy()
                n = X.shape[1]
                mean = np.mean(h, axis=1)
                var = np.var(h, axis=1)
                self.mu = mean
                self.v = var
            if set_average:
                self.mu_average = mean
                self.v_average = var
            h = np.diag(1 / np.sqrt(var+1e-6)) @ (h.T - mean).T
            self.s_hat = h.copy()
            h = ((h.T * self.gamma) + self.beta).T

        if self.after and self.use_bn:
            self.output = h
            return h
        if self.activation == 'relu':
            self.output = relu(h)
        else:
            self.output = softmax(h)
        return self.output

    def _batch_norm_back(self, g_in):
        n = g_in.shape[1]
        sigma1 = 1 / np.sqrt(self.v+1e-6)
        sigma2 = 1 / np.power(self.v+1e-6, 1.5)
        g1 = (g_in.T * sigma1).T
        g2 = (g_in.T * sigma2).T
        d = (self.s.T - self.mu).T
        c = (g2 * d) @ np.ones([n, 1])
        return g1 - (((g1 @ np.ones([n, 1])) @ np.ones([1, n])) / n) - d * (c @ np.ones([1, n])) / n

    def backward(self, g_in):
        n_batch = self.input.shape[1]
        self.gradient = [None, None, None, None]
        if self.activation == 'softmax':
            gradient = g_in
            g_W = gradient @ self.input.T / n_batch + 2 * self.lambda_ * self.W
            g_b = np.mean(gradient, axis=1)
            self.gradient[0] = g_W
            self.gradient[1] = g_b

        elif self.activation == 'relu':
            gradient = g_in
            if not self.after:
                gradient = np.where(self.output > 0, gradient, 0)
            if self.use_bn:
                g_gamma = np.mean(gradient * self.s_hat, axis=1)
                g_beta = np.mean(gradient, axis=1)
                self.gradient[2] = g_gamma
                self.gradient[3] = g_beta
                n = gradient.shape[1]
                gradient = (gradient.T * self.gamma).T
                gradient = self._batch_norm_back(gradient)
            if self.after:
                gradient = np.where(self.s > 0, gradient, 0)
            g_W = gradient @ self.input.T / n_batch + 2 * self.lambda_ * self.W
            g_b = np.mean(gradient, axis=1)
            self.gradient[0] = g_W
            self.gradient[1] = g_b

        return self.W.T @ gradient

    def update(self, leraning_rate):
        if type(self.gradient[0]) == type(None) or type(self.gradient[1]) == type(None):
            raise Exception('First you must calculate gradient')
        if not self.adam:
            self.W -= leraning_rate * self.gradient[0]
            self.b -= leraning_rate * self.gradient[1]
            if self.use_bn:
                self.gamma -= leraning_rate * self.gradient[2]
                self.beta -= leraning_rate * self.gradient[3]
                self.mu_average = self.alpha * \
                    self.mu_average + (1-self.alpha) * self.mu
                self.v_average = self.alpha * \
                    self.v_average + (1-self.alpha) * self.v
        if self.adam:
            m_hat = [0, 0, 0, 0]
            v_hat = [0, 0, 0, 0]
            for i in range(4 if self.use_bn else 2):
                self.adam_m[i] = 0.9 * self.adam_m[i] + 0.1 * self.gradient[i]
                self.adam_v[i] = 0.999 * self.adam_v[i] + \
                    0.001 * np.square(self.gradient[i])
                m_hat[i] = self.adam_m[i] / (1 - (0.9)**self.t)
                v_hat[i] = self.adam_v[i] / (1 - (0.999)**self.t)
            self.W -= leraning_rate * m_hat[0] / (np.sqrt(v_hat[0])+1e-8)
            self.b -= leraning_rate * m_hat[1] / (np.sqrt(v_hat[1])+1e-8)
            if self.use_bn:
                self.gamma -= leraning_rate * \
                    m_hat[2] / (np.sqrt(v_hat[2])+1e-8)
                self.beta -= leraning_rate * \
                    m_hat[3] / (np.sqrt(v_hat[3])+1e-8)
                self.mu_average = self.alpha * \
                    self.mu_average + (1-self.alpha) * self.mu
                self.v_average = self.alpha * \
                    self.v_average + (1-self.alpha) * self.v
        self.t += 1
        self.gradient = [None, None, None, None]


class Model:
    def __init__(self, n_in, number_of_units, n_out, lambda_, use_bn, after=False, adam=False) -> None:
        if type(number_of_units) != list:
            raise Exception('number of laters per units must be a list!')
        self.layers = []
        if len(number_of_units) == 0:
            self.layers.append(Dense_layer(
                n_in, n_out, 'softmax', lambda_, False, adam=adam))
        else:
            for n in number_of_units:
                self.layers.append(Dense_layer(
                    n_in, n, 'relu', lambda_, use_bn, after, adam=adam))
                n_in = n
            self.layers.append(Dense_layer(
                n_in, n_out, 'softmax', lambda_, False, adam=adam))
        self.use_bn = use_bn

    def _forward(self, X, is_test):
        x = X
        for l in self.layers:
            x = l.forward(x, is_test)
        return x

    def set_average(self, X):
        x = X
        for l in self.layers:
            x = l.forward(x, False, True)
        return x

    def _backward(self, Y, P):
        G = -(Y - P)
        for l in self.layers[::-1]:
            G = l.backward(G)

    def _update(self, learning_rate):
        for l in self.layers:
            l.update(learning_rate)

    def _loss(self, Y, P):
        minus_log = -1 * np.log(np.sum(P * Y, axis=0))
        return minus_log.mean()

    def _cost(self, Y, P):
        c = self._loss(Y, P)
        for l in self.layers:
            c += l.lambda_ * np.sum(np.square(l.W))
        return c

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

    def _numerical_gradient(self, X, Y):
        h = 1e-6
        G_Ws = []
        G_bs = []
        G_gammas = []
        G_betas = []
        c = self._cost(Y, self._forward(X, False))
        for idx, l in enumerate(self.layers):
            g_w = np.zeros(l.W.shape)
            g_b = np.zeros(l.b.shape)
            w_save = l.W
            b_save = l.b
            for i in range(len(l.b)):
                l.b = b_save.copy()
                l.b[i] += h
                c2 = self._cost(Y, self._forward(X, False))
                g_b[i] = (c2-c) / h
            l.b = b_save
            for i in range(l.W.shape[0]):
                for j in range(l.W.shape[1]):
                    l.W = w_save.copy()
                    l.W[i][j] += h
                    c2 = self._cost(Y, self._forward(X, False))
                    g_w[i][j] = (c2-c) / h
            l.W = w_save
            G_Ws.append(g_w)
            G_bs.append(g_b)
            if self.use_bn and idx < len(self.layers)-1:
                g_gamma = np.zeros(l.gamma.shape)
                g_beta = np.zeros(l.beta.shape)
                gamma_save = l.gamma
                for i in range(len(l.gamma)):
                    l.gamma = gamma_save.copy()
                    l.gamma[i] += h
                    c2 = self._cost(Y, self._forward(X, False))
                    g_gamma[i] = (c2 - c) / h
                l.gamma = gamma_save
                beta_save = l.beta
                for i in range(len(l.beta)):
                    l.beta = beta_save.copy()
                    l.beta[i] += h
                    c2 = self._cost(Y, self._forward(X, False))
                    g_beta[i] = (c2 - c) / h
                l.beta = beta_save
                G_gammas.append(g_gamma)
                G_betas.append(g_beta)

        return G_Ws, G_bs, G_gammas, G_betas

    def gradient_tester(self, X, Y):
        saves = []
        for l in self.layers:
            saves.append((l.W, l.b))
            l.W = np.random.normal(0, 2, l.W.shape)
            l.b = np.random.normal(0, 2, l.b.shape)
        G_Ws, G_bs, G_gammas, G_betas = self._numerical_gradient(X, Y)
        self._backward(Y, self._forward(X, False))
        for i, l in enumerate(self.layers):
            print(f'layer {i}:')
            diff = G_Ws[i] - l.gradient[0]
            print(
                f'\tW: mean={l.gradient[0].mean()} std={l.gradient[0].std()} mean_diff={diff.mean()} std_diff={diff.std()} max_diff={diff.max()}')
            diff = G_bs[i] - l.gradient[1]
            print(
                f'\tb: mean={l.gradient[1].mean()} std={l.gradient[1].std()} mean_diff={diff.mean()} std_diff={diff.std()} max_diff={diff.max()}')
            if self.use_bn and i < len(G_Ws)-1:
                diff = G_gammas[i] - l.gradient[2]
                print(
                    f'\tgamma: mean={l.gradient[2].mean()} std={l.gradient[2].std()} mean_diff={diff.mean()} std_diff={diff.std()} max_diff={diff.max()}')
                diff = G_betas[i] - l.gradient[3]
                print(
                    f'\tbetas: mean={l.gradient[3].mean()} std={l.gradient[3].std()} mean_diff={diff.mean()} std_diff={diff.std()} max_diff={diff.max()}')

            print('-----------')
            l.W = saves[i][0]
            l.b = saves[i][1]

    def evaluate(self, X, Y):
        P = self._forward(X, True)
        return [self._loss(Y, P), self._cost(Y, P), self._accuracy(Y, P)]

    def fit(self, X, Y, learning_rate, epochs, batch_size, x_val, y_val, lr_scheduler=None, plot_iter=10):
        n = X.shape[1]
        logs = {'train_loss': [], 'train_acc': [], 'train_cost': []}
        logs['val_loss'] = []
        logs['val_acc'] = []
        logs['val_cost'] = []
        iter = 0
        for e in range(epochs):
            X, Y = self._shuffle(X, Y)
            print(f'epoch{e+1}/{epochs}')
            with tqdm(range(0, (n//batch_size)+1)) as pbar:
                for i in pbar:
                    start_index, end_index = i*batch_size, (i+1)*batch_size
                    if i*batch_size == min(n, end_index):
                        break
                    x = X[:, start_index: end_index]
                    y = Y[:, start_index: end_index]
                    p = self._forward(x, False)
                    self._backward(y, p)
                    # pbar.set_description(f'loss:{self._cost(y, p):.3f}')
                    if type(lr_scheduler) != type(None):
                        learning_rate = lr_scheduler(iter, learning_rate)
                    self._update(learning_rate)
                    if i % plot_iter == 0:
                        loss_val, cost_val, acc_val = self.evaluate(
                            x_val, y_val)
                        loss_train, cost_train, acc_train = self.evaluate(X, Y)
                        logs['train_loss'].append(loss_train)
                        logs['train_cost'].append(cost_train)
                        logs['train_acc'].append(acc_train)
                        logs['val_loss'].append(loss_val)
                        logs['val_cost'].append(cost_val)
                        logs['val_acc'].append(acc_val)
                    iter += 1
            loss_val, cost_val, acc_val = self.evaluate(
                x_val, y_val)
            loss_train, cost_train, acc_train = self.evaluate(X, Y)
            logs['train_loss'].append(loss_train)
            logs['train_cost'].append(cost_train)
            logs['train_acc'].append(acc_train)
            logs['val_loss'].append(loss_val)
            logs['val_cost'].append(cost_val)
            logs['val_acc'].append(acc_val)
        return logs


if __name__ == '__main__':
    x_train, x_val, x_test, y_train, y_val, y_test, mean, std = get_all_data(
        os.path.join(os.pardir, 'Data', 'cifar-10-batches-py'))

    # model = MLP(x_train.shape[0], 50, 10, 0.0)
    # # p, h = model._forward(x_train)
    # # model._backward(x_train, y_train, p, h)
    # # model.gradient_checker(x_train[:, :30], y_train[:, :30], ComputeGradsNum)
    # model.fit(x_train[:, :100], y_train[:, :100], 0.01, 200, 10, x_val, y_val)

    model = Model(x_train.shape[0], [10, 10], 10, 0.0, True)
    logs = model.fit(x_train, y_train, 0.01, 12, 20, x_val, y_val)
    plt.plot(logs['val_loss'])
    print(model.evaluate(x_test, y_test))
    plt.show()
    # model.gradient_tester(x_val[:, : 10], y_val[:, : 10])
