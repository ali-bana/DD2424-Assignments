
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os


def _unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


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


def relu(logits):
    return np.where(logits > 0, logits, 0)


def _load_batch(file_path):
    loaded = _unpickle(file_path)
    return loaded[b'data'], loaded[b'labels']


def _load_files(file_names, one_hot=True):
    X, Y = [], []
    for f in file_names:
        loaded = _load_batch(f)
        X.append(loaded[0])
        Y.append(loaded[1])
    if one_hot:
        return np.concatenate(X), one_hot_encoder(np.concatenate(Y))
    return np.concatenate(X), np.concatenate(Y)


def one_hot_encoder(y, num_classes=10):
    return np.eye(num_classes)[np.array(y, dtype='int')]


def _load_data(train_files, validation_files, test_files):
    x_train = []
    x_val = []
    x_test = []
    y_train = []
    y_val = []
    y_test = []

    x_train, y_train = _load_files(train_files)
    x_val, y_val = _load_files(validation_files)
    x_test, y_test = _load_files(test_files)

    mean = 0
    std = 0
    mean = x_train.mean()
    std = x_train.std()
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    x_val = (x_val - mean) / std
    return x_train.T, x_val.T, x_test.T, y_train.T, y_val.T, y_test.T, mean, std


def get_data(data_folder):
    train_files = [os.path.join(data_folder, 'cifar-10-batches-py', f)
                   for f in ['data_batch_1']]
    val_files = [os.path.join(data_folder, 'cifar-10-batches-py', f)
                 for f in ['data_batch_2']]
    test_files = [os.path.join(data_folder, 'cifar-10-batches-py', f)
                  for f in ['test_batch']]
    return _load_data(train_files, val_files, test_files)


def flip(X):
    X2 = X.copy()
    for i in range(32 * 3 - 1):
        X2[i*32:(i+1)*32] = X2[i*32:(i+1)*32][::-1]
    return X2


def get_all_data(dir_path, validation_n=5000):
    x_train = []
    x_val = []
    x_test = []
    y_train = []
    y_val = []
    y_test = []

    for i in range(1, 6):
        file_path = os.path.join(dir_path, f'data_batch_{i}')
        loaded = _unpickle(file_path)
        x_train.append(loaded[b'data'])
        y_train.append(loaded[b'labels'])
    loaded = _unpickle(os.path.join(dir_path, 'test_batch'))
    x_test = np.array(loaded[b'data'])
    y_test = np.eye(10)[np.array(loaded[b'labels'])]  # one-hot coding
    x_train = np.concatenate(x_train)
    y_train = np.eye(10)[np.concatenate(y_train)]  # one-hot coding
    mean = 0
    std = 0
    mean = x_train.mean()
    std = x_train.std()
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    x_val = x_train[:validation_n]
    y_val = y_train[:validation_n]
    x_train = x_train[validation_n:]
    y_train = y_train[validation_n:]
    return x_train.T, x_val.T, x_test.T, y_train.T, y_val.T, y_test.T, mean, std


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


n_in = 3072
n_out = 10
n_batch = 100


def make_circular_scheduler(max_lr, min_lr, n_s, decay_max=False, decay_factor=0.8):
    def scheduler(iter, lr):
        nonlocal max_lr
        quotient = iter // n_s
        if decay_max and iter > 0 and iter % (2*n_s) == 0:
            max_lr = max_lr * decay_factor
        if quotient % 2 == 0:  # increasing part
            return min_lr + (iter - quotient*n_s) * (max_lr - min_lr) / n_s
        else:  # decreasing part
            return max_lr - (iter - quotient*n_s) * (max_lr - min_lr) / n_s
    return scheduler


def logs_plotter(logs, name, factor=10):
    n_epochs = len(logs['train_loss'])
    plt.plot([i*factor+1 for i in range(n_epochs)],
             logs['train_loss'], label='train loss')
    plt.plot([i*factor+1 for i in range(n_epochs)],
             logs['val_loss'], label='validation loss')
    plt.legend()
    plt.title('Loss over training')
    plt.savefig(f'results/{name}_loss.png')
    plt.close()

    plt.plot([i*factor+1 for i in range(n_epochs)],
             logs['train_cost'], label='train cost')
    plt.plot([i*10+1 for i in range(n_epochs)],
             logs['val_cost'], label='validation cost')
    plt.legend()
    plt.title('Cost over training')
    plt.savefig(f'results/{name}_cost.png')
    plt.close()

    plt.plot([i*factor+1 for i in range(n_epochs)],
             logs['train_acc'], label='train accuracy')
    plt.plot([i*factor+1 for i in range(n_epochs)],
             logs['val_acc'], label='validation accuracy')
    plt.legend()
    plt.title('Accuracy over training')
    plt.savefig(f'results/{name}_acc.png')
    plt.close()


def test_gradient():
    x_train, x_val, x_test, y_train, y_val, y_test, mean, std = get_all_data(
        os.path.join(os.pardir, 'Data', 'cifar-10-batches-py'))

    model = Model(10, [5, 5], n_out, 0.01, False)
    print('model with no batch norm')
    model.gradient_tester(x_train[:10, :10], y_train[:10, :10])
    model = Model(10, [5, 5], n_out, 0.01, True)
    print('model with batch norm')
    model.gradient_tester(x_train[:10, :10], y_train[:10, :10])


def exercise_2_1():
    x_train, x_val, x_test, y_train, y_val, y_test, mean, std = get_all_data(
        os.path.join(os.pardir, 'Data', 'cifar-10-batches-py'))
    x_train, y_train = x_train[:45000], y_train[:45000]

    model = Model(n_in, [50, 50], n_out, 0.005, False)
    scheduler = make_circular_scheduler(0.1, 1e-5, 5*45000/n_batch)
    logs = model.fit(x_train, y_train, 0.01, 20,
                     n_batch, x_val, y_val, scheduler)
    logs_plotter(logs, 'ex2_no_batch')
    with open('results/ex2.txt', 'a') as f:
        f.write(
            f'no_batch_norm: test_set:{model.evaluate(x_test, y_test)}, val_test:{model.evaluate(x_val, y_val)}\n')

    model = Model(n_in, [50, 50], n_out, 0.005, True)
    scheduler = make_circular_scheduler(0.1, 1e-5, 5*45000/n_batch)
    logs = model.fit(x_train, y_train, 0.01, 20,
                     n_batch, x_val, y_val, scheduler)
    logs_plotter(logs, 'ex2_with_batch')
    with open('results/ex2.txt', 'a') as f:
        f.write(
            f'with_batch_norm: test_set:{model.evaluate(x_test, y_test)}, val_test:{model.evaluate(x_val, y_val)}\n')


def exersice_2_2():
    x_train, x_val, x_test, y_train, y_val, y_test, mean, std = get_all_data(
        os.path.join(os.pardir, 'Data', 'cifar-10-batches-py'))
    x_train, y_train = x_train[:45000], y_train[:45000]

    model = Model(n_in, [50, 30, 20, 20, 10, 10, 10], n_out, 0.005, False)
    scheduler = make_circular_scheduler(0.1, 1e-5, 5*45000/n_batch)
    logs = model.fit(x_train, y_train, 0.01, 20,
                     n_batch, x_val, y_val, scheduler)
    logs_plotter(logs, 'ex2_9layer_no_batch')
    with open('results/ex2.txt', 'a') as f:
        f.write(
            f'no_batch_norm 9layer: test_set:{model.evaluate(x_test, y_test)}, val_test:{model.evaluate(x_val, y_val)}\n')

    model = Model(n_in, [50, 30, 20, 20, 10, 10, 10], n_out, 0.005, True)
    scheduler = make_circular_scheduler(0.1, 1e-5, 5*45000/n_batch)
    logs = model.fit(x_train, y_train, 0.01, 20,
                     n_batch, x_val, y_val, scheduler)
    logs_plotter(logs, 'ex2_with_batch_9layer')
    with open('results/ex2.txt', 'a') as f:
        f.write(
            f'with_batch_norm 9layer: test_set:{model.evaluate(x_test, y_test)}, val_test:{model.evaluate(x_val, y_val)}\n')


def initial_sensitivity():
    x_train, x_val, x_test, y_train, y_val, y_test, mean, std = get_all_data(
        os.path.join(os.pardir, 'Data', 'cifar-10-batches-py'))
    x_train, y_train = x_train[:45000], y_train[:45000]
    scheduler = make_circular_scheduler(0.1, 1e-5, 2*45000/n_batch)
    for sigma in [1e-1, 1e-3, 1e-4]:
        for use_bn in [True, False]:
            model = Model(n_in, [50, 50], n_out, 0.005, use_bn)
            # chainging the weights
            for l in model.layers:
                l.W = np.random.normal(0, sigma, l.W.shape)
            logs = model.fit(x_train, y_train, 0.01, 8,
                             n_batch, x_val, y_val, scheduler)
            logs_plotter(logs, f'sens_sigma_{sigma}_use_bn_{use_bn}')
            with open('results/weight_sens.txt', 'a') as f:
                f.write(
                    f'sigma={sigma} use_bn={use_bn}, result_on_test={model.evaluate(x_test, y_test)}\n')


def lambda_search():
    x_train, x_val, x_test, y_train, y_val, y_test, mean, std = get_all_data(
        os.path.join(os.pardir, 'Data', 'cifar-10-batches-py'))
    x_train, y_train = x_train[:45000], y_train[:45000]
    scheduler = make_circular_scheduler(0.1, 1e-5, 2*45000/n_batch)
    for lambda_ in [0.1, 0.05, 0.01, 0.005, 0.001, 0.0001, 0.0]:
        model = Model(n_in, [50, 50], n_out, lambda_, True)
        model.fit(x_train, y_train, 0.01, 8,
                  n_batch, x_val, y_val, scheduler)
        with open('results/lambda_search.txt', 'a') as f:
            f.write(
                f'lambda={lambda_}  result_on_test={model.evaluate(x_test, y_test)}, result_on_val={model.evaluate(x_val, y_val)}\n')


def average_set():
    x_train, x_val, x_test, y_train, y_val, y_test, mean, std = get_all_data(
        os.path.join(os.pardir, 'Data', 'cifar-10-batches-py'))
    x_train, y_train = x_train[:45000], y_train[:45000]
    model = Model(n_in, [50, 50], n_out, 0.005, True)
    scheduler = make_circular_scheduler(0.1, 1e-5, 5*45000/n_batch)
    logs = model.fit(x_train, y_train, 0.01, 20,
                     n_batch, x_val, y_val, scheduler)
    model.set_average(x_train[:10000])
    with open('results/set_average.txt', 'a') as f:
        f.write(
            f'no_batch_norm: test_set:{model.evaluate(x_test, y_test)}, val_test:{model.evaluate(x_val, y_val)}\n')


def adaptive_bn():
    x_train, x_val, x_test, y_train, y_val, y_test, mean, std = get_all_data(
        os.path.join(os.pardir, 'Data', 'cifar-10-batches-py'))
    x_train, y_train = x_train[:45000], y_train[:45000]
    model = Model(n_in, [50, 50], n_out, 0.005, True)
    scheduler = make_circular_scheduler(0.1, 1e-5, 5*45000/n_batch)
    model.fit(x_train, y_train, 0.01, 20,
              n_batch, x_val, y_val, scheduler)
    flipped_test = flip(x_test)
    length = 10000
    condition = np.ones((n_in, length)) * \
        np.random.binomial(1, 0.5, length)
    x_avg = np.where(
        condition > 0.5, x_test[:, :length], flipped_test[:, :length])
    model.set_average(x_avg)
    with open('results/adaptive.txt', 'a') as f:
        f.write(
            f'test_set:{model.evaluate(x_test, y_test)}, val_test:{model.evaluate(x_val, y_val)}\n')


def after_tester():
    x_train, x_val, x_test, y_train, y_val, y_test, mean, std = get_all_data(
        os.path.join(os.pardir, 'Data', 'cifar-10-batches-py'))
    x_train, y_train = x_train[:45000], y_train[:45000]
    model = Model(n_in, [50, 50], n_out, 0.005, True, True)
    scheduler = make_circular_scheduler(0.1, 1e-5, 5*45000/n_batch)
    logs = model.fit(x_train, y_train, 0.01, 20,
                     n_batch, x_val, y_val, scheduler)
    with open('results/after.txt', 'a') as f:
        f.write(
            f'after: test_set:{model.evaluate(x_test, y_test)}, val_test:{model.evaluate(x_val, y_val)}\n')


def adam():
    x_train, x_val, x_test, y_train, y_val, y_test, mean, std = get_all_data(
        os.path.join(os.pardir, 'Data', 'cifar-10-batches-py'))
    x_train, y_train = x_train[:45000], y_train[:45000]
    model = Model(n_in, [50, 50], n_out, 0.005, True, False, True)
    scheduler = make_circular_scheduler(0.1, 1e-5, 5*45000/n_batch, True, 0.6)
    logs = model.fit(x_train, y_train, 0.01, 30,
                     n_batch, x_val, y_val, scheduler)
    with open('results/adam.txt', 'a') as f:
        f.write(
            f'adam: test_set:{model.evaluate(x_test, y_test)}, val_test:{model.evaluate(x_val, y_val)}\n')


def more_hidden():
    x_train, x_val, x_test, y_train, y_val, y_test, mean, std = get_all_data(
        os.path.join(os.pardir, 'Data', 'cifar-10-batches-py'))
    x_train, y_train = x_train[:45000], y_train[:45000]
    model = Model(n_in, [50, 30, 30, 30, 20, 20, 20, 20],
                  n_out, 0.005, True, False, False)
    scheduler = make_circular_scheduler(0.1, 1e-5, 5*45000/n_batch)
    logs = model.fit(x_train, y_train, 0.01, 20,
                     n_batch, x_val, y_val, scheduler)
    with open('results/bigger.txt', 'a') as f:
        f.write(
            f'bigger: test_set:{model.evaluate(x_test, y_test)}, val_test:{model.evaluate(x_val, y_val)}\n')


def main():
    # test_gradient()
    # exercise_2()
    # exersice_2_2()
    # initial_sensitivity()
    # lambda_search()
    # average_set()
    # adaptive_bn()
    # after_tester()
    # adam()
    more_hidden()


if __name__ == '__main__':
    main()
