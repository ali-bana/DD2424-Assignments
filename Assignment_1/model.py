import numpy as np
from tqdm import tqdm
from utils import load_data
import os


class Linear_layer:
    def __init__(self, input_dim, output_dim, lamda) -> None:
        self.W = np.random.normal(0, 0.01, size=(input_dim, output_dim))
        self.bias = np.random.normal(0, 0.01, size=output_dim)
        self.X = None
        self.g_W = None
        self.g_bias = None
        self.lamda = lamda

    def forward(self, X):
        self.X = X
        ret = X @ self.W + self.bias
        return ret

    def calculate_gradient(self, g_s):
        dl_db = g_s
        n = g_s.shape[0]
        input_dim = (self.W.shape)[0]
        w_broadcasted = np.repeat(self.W[None, :], n, axis=0)
        g_broadcasted = np.repeat(g_s[:, None, :], input_dim, axis=1)
        dl_dw = g_broadcasted * w_broadcasted
        self.g_bias = dl_db.mean(axis=0)
        self.g_W = dl_dw.mean(axis=0) + 2*self.lamda*self.W

    def update(self, lr):
        self.W -= lr * self.g_W
        self.bias -= lr * self.g_bias


class Softmax_layer:
    def __init__(self) -> None:
        self.probs = None

    def forward(self, logits):
        assert len(logits.shape) == 2
        s = np.max(logits, axis=1)
        s = s[:, np.newaxis]
        e_x = np.exp(logits - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis]
        probs = e_x / div
        probs += np.ones(probs.shape) * 1e-6
        self.probs = probs
        return probs

    def calculate_gradient(self, y):
        return self.probs - y

    def update(self, lr): pass


class Model:
    def __init__(self, input_dim, output_dim, lamda) -> None:
        self.layers = [Linear_layer(
            input_dim, output_dim, lamda), Softmax_layer()]
        self.lamda = lamda

    def forward(self, X):
        x = X
        for l in self.layers:
            x = l.forward(x)
        return x

    def calculate_gradient(self, y):
        g = y
        for l in self.layers[::-1]:
            g = l.calculate_gradient(g)

    def compute_cost(self, X, y):
        probs = self.forward(X)
        minus_log = -1 * np.log(np.sum(probs * y, axis=1))
        return minus_log.mean() + self.lamda * np.sum(np.square(self.layers[0].W))

    def _loss(self, Y, P):
        minus_log = -1 * np.log(np.sum(P * Y, axis=1))
        return minus_log.mean() + self.lamda * np.sum(np.square(self.layers[0].W))

    def _accuracy(self, Y, P):
        y_num = np.argmax(Y, axis=1)
        y_pred = np.argmax(P, axis=1)
        return np.where(y_num == y_pred, 1, 0).sum() / y_num.shape[0]

    def _update(self, lr):
        for l in self.layers:
            l.update(lr)

    def _evaluation_str(self, X, Y):
        p = self.forward(X)
        return f'accuracy={round(self._accuracy(Y, p),2)}, loss={round(self._loss(Y, p), 2)}'

    def fit(self, X, Y, learning_rate, batch_size, epoch, x_val, y_val):
        n = X.shape[0]
        for e in range(epoch):
            with tqdm(range(0, n//batch_size+1)) as pbar:
                for i in pbar:
                    if i*batch_size == min(n, (i+1)*batch_size):
                        break
                    x = X[i*batch_size: min(n, (i+1)*batch_size)]
                    y = Y[i*batch_size: min(n, (i+1)*batch_size)]
                    p = self.forward(x)
                    if i % 100 == 0:
                        pbar.set_description(
                            f'{e}/{epoch}, accuracy={round(self._accuracy(y, p),2)}, loss={round(self._loss(y, p), 2)}')
                    self.calculate_gradient(y)
                    self._update(learning_rate)

                p = self.forward(x_val)
                print(
                    f'validation:{self._evaluation_str(x_val, y_val)} | training:{self._evaluation_str(x_train, y_train)}')


if __name__ == '__main__':
    x_train, x_val, x_test, y_train, y_val, y_test = load_data(
        os.path.join(os.getcwd(), 'Data', 'cifar-10-batches-py'))

    model = Model(3072, 10, 0)
    model.fit(x_train, y_train, 0.01, 100, 20, x_val, y_val)
