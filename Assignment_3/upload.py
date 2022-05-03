import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os


def _unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def display_flat_image(img):
    img += -1 * img.min()
    img /= img.max()
    reshaped = img.reshape([3, 32, 32])
    r = reshaped[0, :, :]
    g = reshaped[1, :, :]
    b = reshaped[2, :, :]
    r -= r.min()
    g -= g.min()
    b -= g.min()
    r /= r.max()
    g /= g.max()
    b /= b.max()
    plt.imshow(np.dstack((r, g, b)))
    plt.show()


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


n_in = 3072
n_out = 10


class MLP:
    def __init__(self, n_in, n_h, n_out, lambda_, dropout=-1) -> None:
        self.W1 = np.random.normal(0, 1/np.sqrt(n_h), (n_h, n_in))
        self.b1 = np.zeros(n_h)
        self.W2 = np.random.normal(0, 1/np.sqrt(n_out), (n_out, n_h))
        self.b2 = np.zeros(n_out)
        self.lambda_ = lambda_
        self.drop_out = -1

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

    def _forward(self, X, testing=False):
        h = self._relu(((self.W1 @ X).T + self.b1).T)
        if not testing and self.drop_out != -1:
            mask = np.random.binomial(1, self.drop_out, h.shape)
            print(self.mask)
            h = h * mask
        elif testing and self.drop_out != -1:
            h = h * self.drop_out
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

    def fit(self, X, Y, learning_rate, epochs, batch_size, x_val, y_val, lr_scheduler=None, plot_iter=10, flipping=False):
        n = X.shape[1]
        logs = {'train_loss': [], 'train_acc': [], 'train_cost': []}
        logs['val_loss'] = []
        logs['val_acc'] = []
        logs['val_cost'] = []
        iter = 0
        if flipping:
            X_flipped = flip(X)
        for e in range(epochs):
            if flipping:
                X, Y, X_flipped = self._shuffle(X, Y, X_flipped)
            else:
                X, Y = self._shuffle(X, Y)

            with tqdm(range(0, (n//batch_size)+1)) as pbar:
                for i in pbar:
                    start_index, end_index = i*batch_size, (i+1)*batch_size
                    if i*batch_size == min(n, end_index):
                        break
                    x = X[:, start_index: end_index]
                    y = Y[:, start_index: end_index]
                    if flipping:
                        x_not_fliped = X[:, start_index: end_index]
                        x_flipped = X_flipped[:, start_index: end_index]
                        length = end_index - start_index
                        condition = np.ones((x.shape[0], length)) * \
                            np.random.binomial(1, 0.5, length)
                        x = np.where(condition > 0.5, x_flipped, x_not_fliped)
                    p, h = self._forward(x)
                    g_W1, g_b1, g_W2, g_b2 = self._backward(x, y, p, h)
                    if type(lr_scheduler) != type(None):
                        learning_rate = lr_scheduler(iter, learning_rate)
                    self.W1 -= learning_rate * g_W1
                    self.W2 -= learning_rate * g_W2
                    self.b1 -= learning_rate * g_b1
                    self.b2 -= learning_rate * g_b2
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

            print(
                f'epoch {e+1}/{epochs}\n\tTraining: loss:{loss_train}, cost{cost_train}, acc:{acc_train}')
            print(
                f'\tValidation: loss:{loss_val}, cost{cost_val}, acc:{acc_val}')
        return logs

    def evaluate(self, X, Y):
        P, H = self._forward(X)
        return [self._loss(Y, P), self._cost(Y, P), self._accuracy(Y, P)]


input_size = 3072


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


def lambda_coarse():
    x_train, x_val, x_test, y_train, y_val, y_test, mean, std = get_all_data(os.path.join(
        os.getcwd(), 'Data', 'cifar-10-batches-py'))
    batch_size = 100
    n_s = 2 * round(x_train.shape[1] / batch_size)
    for lambda_ in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]:
        scheduler = make_circular_scheduler(1e-1, 1e-5, n_s)
        model = MLP(input_size, 50, 10, lambda_)
        logs = model.fit(x_train, y_train, 0, 4, batch_size,
                         x_val, y_val, scheduler)
        with open('results/coarse.txt', 'a') as f:
            val_acc = logs['val_acc'][-1]
            val_loss = logs['val_loss'][-1]
            f.write(f'lambda: {lambda_}, acc: {val_acc}, loss:{val_loss}\n')


def lambda_fine(good_lambda):
    x_train, x_val, x_test, y_train, y_val, y_test, mean, std = get_all_data(os.path.join(
        os.getcwd(), 'Data', 'cifar-10-batches-py'))
    batch_size = 100
    n_s = 2 * round(x_train.shape[1] / batch_size)
    for lambda_ in np.random.uniform(0.5*good_lambda, 1.5*good_lambda, 10):
        scheduler = make_circular_scheduler(1e-1, 1e-5, n_s)
        model = MLP(input_size, 50, 10, lambda_)
        logs = model.fit(x_train, y_train, 0, 8, batch_size,
                         x_val, y_val, scheduler)
        with open('results/fine.txt', 'a') as f:
            val_acc = logs['val_acc'][-1]
            val_loss = logs['val_loss'][-1]
            f.write(f'lambda: {lambda_}, acc: {val_acc}, loss:{val_loss}\n')


def final_model():
    x_train, x_val, x_test, y_train, y_val, y_test, mean, std = get_all_data(os.path.join(
        os.getcwd(), 'Data', 'cifar-10-batches-py'), validation_n=1000)
    model = MLP(input_size, 50, 10, 0.009)
    scheduler = make_circular_scheduler(0.1, 1e-5, 490)
    logs = model.fit(x_train, y_train, 0, 30, 100, x_val, y_val, scheduler)
    logs_plotter(logs, 'final')
    results = model.evaluate(x_test, y_test)
    with open('results/final.txt', 'a') as f:
        f.write(str(results))


def more_nodes():
    x_train, x_val, x_test, y_train, y_val, y_test, mean, std = get_all_data(os.path.join(
        os.getcwd(), 'Data', 'cifar-10-batches-py'))
    batch_size = 100
    n_s = 2 * round(x_train.shape[1] / batch_size)
    for hidden, lambda_ in itertools.product([10, 50, 70, 100, 150], [0.0001, 0.001, 0.01, 0.1]):
        model = MLP(input_size, hidden, 10, lambda_)
        scheduler = make_circular_scheduler(0.1, 1e-5, 490)
        logs = model.fit(x_train, y_train, 0, 8, batch_size,
                         x_val, y_val, scheduler)
        with open('results/more_nodes.txt', 'a') as f:
            val_acc = logs['val_acc'][-1]
            val_loss = logs['val_loss'][-1]
            f.write(f'lambda: {lambda_}, acc: {val_acc}, loss:{val_loss}\n')


def train_big_model():
    x_train, x_val, x_test, y_train, y_val, y_test, mean, std = get_all_data(os.path.join(
        os.getcwd(), 'Data', 'cifar-10-batches-py'))
    batch_size = 450
    scheduler = make_circular_scheduler(0.1, 1e-5, 800, True)
    model = MLP(input_size, 120, 10, 0.01, 0.7)
    logs = model.fit(x_train, y_train, 0.01, 30, batch_size,
                     x_val, y_val, scheduler, flipping=True)

    logs_plotter(logs, '100_node_model')


def optimize_params():
    x_train, x_val, x_test, y_train, y_val, y_test, mean, std = get_all_data(os.path.join(
        os.getcwd(), 'Data', 'cifar-10-batches-py'))
    batch_size = 100
    for decay_factor in [0.5, 0.65, 0.8, 0.9]:
        scheduler = make_circular_scheduler(0.1, 1e-5, 800, True, decay_factor)
        model = MLP(input_size, 120, 10, 0.01, 0.7)
        logs = model.fit(x_train, y_train, 0.01, 20, batch_size,
                         x_val, y_val, scheduler, flipping=True)
        with open('results/last.txt', 'a') as f:
            val_acc = logs['val_acc'][-1]
            val_loss = logs['val_loss'][-1]
            f.write(
                f'decay_factor: {decay_factor}, acc: {val_acc}, loss:{val_loss}\n')

    for lammbda_ in [0, 0.001, 0.01]:
        scheduler = make_circular_scheduler(0.1, 1e-5, 800, True, 0.8)
        model = MLP(input_size, 120, 10, lammbda_, 0.7)
        logs = model.fit(x_train, y_train, 0.01, 20, batch_size,
                         x_val, y_val, scheduler, flipping=True)
        with open('results/last.txt', 'a') as f:
            val_acc = logs['val_acc'][-1]
            val_loss = logs['val_loss'][-1]
            f.write(f'lammbda: {lammbda_}, acc: {val_acc}, loss:{val_loss}\n')

    for dropout, lambda_ in itertools.product([1, 0.85, 0.7, 0.55, 0.4], [0, 0.001, 0.01]):
        scheduler = make_circular_scheduler(0.1, 1e-5, 800, True, 0.8)
        model = MLP(input_size, 120, 10, lambda_, dropout)
        logs = model.fit(x_train, y_train, 0.01, 20, batch_size,
                         x_val, y_val, scheduler, flipping=True)
        with open('results/last.txt', 'a') as f:
            val_acc = logs['val_acc'][-1]
            val_loss = logs['val_loss'][-1]
            f.write(
                f'lammbda: {lambda_}, dropout:{dropout} acc: {val_acc}, loss:{val_loss}\n')
    scheduler = make_circular_scheduler(0.1, 1e-5, 450, True, 0.8)
    model = MLP(input_size, 120, 10, 0.01, 0.7)
    logs = model.fit(x_train, y_train, 0.01, 80, batch_size,
                     x_val, y_val, scheduler, flipping=True)
    logs_plotter(logs, 'Final_cycle_tracker')


def main():
    x_train, x_val, x_test, y_train, y_val, y_test, mean, std = get_data(
        os.path.join(os.getcwd(), 'Data'))
    # ========================================================
    # checking the gradient calculation
    # model = MLP(input_shape, 50, 10, 1.0)
    # model.gradient_checker(x_train[:, :30], y_train[:, :30], ComputeGradsNum)
    # =========================================================
    # Making sure the network is learning by overfitting to 100 datapoints
    # model = MLP(input_shape, 50, 10, 0.0)
    # model.fit(x_train[:, :100], y_train[:, :100], 0.01, 200, 10, x_val, y_val)
    # =========================================================
    # checking if the scheduler is working properly
    # scheduler = make_circular_scheduler(0.1, 0.05, 10)
    # plt.plot([scheduler(i, 0) for i in range(100)])
    # plt.savefig('results/scheduler.png')
    # plt.show()
    # =========================================================
    # Exercise 3 train with cyclic
    # scheduler = make_circular_scheduler(0.1, 1e-5, 500)
    # model = MLP(input_size, 50, 10, 0.01)
    # logs = model.fit(x_train, y_train, 0.01, 10, 100, x_val, y_val, scheduler)
    # logs_plotter(logs, 'temp')
    # =========================================================
    # scheduler = make_circular_scheduler(0.1, 1e-5, 800)
    # model = MLP(input_size, 50, 10, 0.01)
    # logs = model.fit(x_train, y_train, 0.01, 48, 100, x_val, y_val, scheduler)
    # logs_plotter(logs, 'exercise4_')
    # =========================================================
    # lambda_coarse()
    # lambda_fine(0.01)
    # =========================================================
    # Last part
    # final_model()
    # =========================================================
    # more hidden nodes
    # more_nodes()
    # =========================================================
    # test with flipping
    # scheduler = make_circular_scheduler(0.1, 1e-5, 800)
    # model = MLP(input_size, 50, 10, 0.01)
    # logs = model.fit(x_train, y_train, 0.01, 48, 100,
    #                  x_val, y_val, scheduler, flipping=True)
    # logs_plotter(logs, 'model_of_ex4_fliping_')
    # =========================================================
    # test drop-out
    # scheduler = make_circular_scheduler(0.1, 1e-5, 800)
    # model = MLP(input_size, 50, 30, 0.01, 0.7)
    # logs = model.fit(x_train, y_train, 0.01, 48, 100,
    #                  x_val, y_val, scheduler)
    # logs_plotter(logs, 'model_of_ex4_dropout_')
    # =========================================================
    # checking if the scheduler decay is working properly
    # scheduler = make_circular_scheduler(0.1, 0.01, 10, True)
    # plt.plot([scheduler(i, 0) for i in range(100)])
    # plt.savefig('results/scheduler_decay.png')
    # plt.show()
    # =========================================================
    # train_big_model()
    # =========================================================
    # optimize_params()


if __name__ == '__main__':
    main()
