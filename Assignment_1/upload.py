import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
n_in = 3072
n_out = 10


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_data(dir_path, validation_n=5000):
    x_train = []
    x_val = []
    x_test = []
    y_train = []
    y_val = []
    y_test = []

    for i in range(1, 6):
        file_path = os.path.join(dir_path, f'data_batch_{i}')
        loaded = unpickle(file_path)
        x_train.append(loaded[b'data'])
        y_train.append(loaded[b'labels'])
    loaded = unpickle(os.path.join(dir_path, 'test_batch'))
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
    y_train += np.random.uniform(low=0, high=0.01, size=y_train.shape)
    y_train = y_train / y_train.sum(axis=1)[:, np.newaxis]
    return x_train.T, x_val.T, x_test.T, y_train.T, y_val.T, y_test.T, mean, std


def shuffle(X, Y, X_shuffled=None):
    n = X.shape[1]
    idx = np.random.permutation(n)
    if type(X_shuffled) == type(None):
        return X[:, idx], Y[:, idx]
    return X[:, idx], Y[:, idx], X_shuffled[:, idx]


def flip(X):
    X2 = X.copy()
    for i in range(32 * 3 - 1):
        X2[i*32:(i+1)*32] = X2[i*32:(i+1)*32][::-1]
    return X2


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


def gradient_checker(X, Y, lamda, slow=False):
    W = np.random.normal(0, 2, size=(n_out, n_in))
    b = np.random.normal(0, 2, (n_out, 1))
    if slow:
        num_g_w, num_g_b = ComputeGradsNumSlow(X, Y, W, b, lamda, 0.0001)
    else:
        num_g_w, num_g_b = ComputeGradsNum(X, Y, W, b, lamda, 0.0001)
    P = forward(X, W, b)
    anal_g_w, anal_g_b = compute_gradients(X, Y, W, b, P, lamda)
    diff_b = np.abs(num_g_b - anal_g_b)
    diff_w = np.abs(num_g_w - anal_g_w)
    # diff = np.abs(numerical_g - analytical_g) / np.where(np.abs(numerical_g) +
    #                                                      np.abs(
    #                                                          analytical_g) < 1e-5, 1e-5,
    #                                                      np.abs(numerical_g)+np.abs(analytical_g))
    print(
        f'For abs of diff of gW: mean: {diff_w.mean()}, std: {diff_w.std()}, max:{diff_w.max()}, gradient min: {num_g_w.min()}, gradient max: {num_g_w.max()}, gradient std: {num_g_w.std()}')
    print(
        f'For abs of diff of gb: mean: {diff_b.mean()}, std: {diff_b.std()}, max:{diff_b.max()}, gradient min: {num_g_b.min()}, gradient max: {num_g_b.max()}, gradient std: {num_g_b.std()}')


def fit_and_plot(x_train, x_val, x_test, y_train, y_val, y_test, lamda, n_epochs, n_batch, eta, flipping=False):
    W = np.random.normal(0, 0.1, size=(n_out, n_in))
    b = np.random.normal(0, 0.1, (n_out, 1))
    W, b, logs = fit(x_train, y_train, W, b, lamda, eta,
                     n_batch, n_epochs, x_val, y_val, flipping)
    plt.plot([i+1 for i in range(n_epochs)],
             logs['train_loss'], label='trainig loss')
    plt.plot([i+1 for i in range(n_epochs)],
             logs['val_loss'], label='validation loss')
    plt.title(
        f'Loss function over training eta={eta}, lambda={lamda}')
    plt.legend()
    plt.savefig(f'results/loss_over_trainig_eta={eta}_lambda={lamda}.png')
    plt.close()

    plt.plot([i+1 for i in range(n_epochs)],
             logs['train_cost'], label='trainig cost')
    plt.plot([i+1 for i in range(n_epochs)],
             logs['val_cost'], label='validation cost')
    plt.title(
        f'Cost function over training eta={eta}, lambda={lamda}')
    plt.legend()
    plt.savefig(f'results/cost_over_trainig_eta={eta}_lambda={lamda}.png')
    plt.close()

    plt.plot([i+1 for i in range(n_epochs)],
             logs['train_acc'], label='trainig accuracy')
    plt.plot([i+1 for i in range(n_epochs)],
             logs['val_acc'], label='validation accuracy')
    plt.title(f'Accuracy over training eta={eta}, lambda={lamda}')
    plt.legend()
    plt.savefig(f'results/accuracy_over_trainig_eta={eta}_lambda={lamda}.png')
    plt.close()

    # evaluation over test set
    eval = evaluate(x_test, y_test, W, b, lamda)
    test_loss = eval['loss']
    test_cost = eval['cost']
    test_acc = eval['acc']
    test_str = f'testing eta={eta}, lambda={lamda}, loss:{test_loss}, cost:{test_cost}, accuracy:{test_acc}'
    print(test_str)
    with open(f'results/eta={eta}_lambda={lamda}.txt', 'a') as f:
        f.write(test_str)
    # for i in range(n_out):
    #     display_flat_image(W[i])

    montage(W, f'results/weights_eta={eta}_lambda={lamda}.png')


def mandatory():
    x_train, x_val, x_test, y_train, y_val, y_test, mean, std = load_data(os.path.join(
        os.getcwd(), 'Data', 'cifar-10-batches-py'))
    gradient_checker(x_train[:, :20], y_train[:, :20], 0.1, True)
    fit_and_plot(x_train, x_val, x_test, y_train,
                 y_val, y_test, 0, 40, 100, 0.1)

    fit_and_plot(x_train, x_val, x_test, y_train,
                 y_val, y_test, 0, 40, 100, 0.001)

    fit_and_plot(x_train, x_val, x_test, y_train,
                 y_val, y_test, 0.1, 40, 100, 0.001)

    fit_and_plot(x_train, x_val, x_test, y_train,
                 y_val, y_test, 1, 40, 100, 0.001)


def bonus_2_1_1():
    x_train2, x_val2, x_test2, y_train2, y_val2, y_test2, mean2, std2 = load_data(os.path.join(
        os.getcwd(), 'Data', 'cifar-10-batches-py'), 1000)

    fit_and_plot(x_train2, x_val2, x_test2, y_train2,
                 y_val2, y_test2, 0, 40, 100, 0.1)

    fit_and_plot(x_train2, x_val2, x_test2, y_train2,
                 y_val2, y_test2, 0, 40, 100, 0.001)

    fit_and_plot(x_train2, x_val2, x_test2, y_train2,
                 y_val2, y_test2, 0.1, 40, 100, 0.001)

    fit_and_plot(x_train2, x_val2, x_test2, y_train2,
                 y_val2, y_test2, 1, 40, 100, 0.001)


def bonus_2_1_2():
    x_train, x_val, x_test, y_train, y_val, y_test, mean, std = load_data(os.path.join(
        os.getcwd(), 'Data', 'cifar-10-batches-py'))
    fit_and_plot(x_train, x_val, x_test, y_train,
                 y_val, y_test, 0, 40, 100, 0.1, True)

    fit_and_plot(x_train, x_val, x_test, y_train,
                 y_val, y_test, 0, 40, 100, 0.001, True)

    fit_and_plot(x_train, x_val, x_test, y_train,
                 y_val, y_test, 0.1, 40, 100, 0.001, True)

    fit_and_plot(x_train, x_val, x_test, y_train,
                 y_val, y_test, 1, 40, 100, 0.001, True)


# def grid_search():
#     batch_size = [16, 64, 128, 256, 512]
#     lamda = [0, 0.001, 0.01, 0.1, 0.5]
#     eta = [0.01, 0.005, 0.001]
#     i = 0
#     for b, l, e in itertools.product(batch_size, lamda, eta):
#         print(i, ':', b, l, e)

#         i += 2

def lr_scheduler(epoch):
    if epoch <= 10:
        return 0.01
    return 0.001 / (10 ** ((epoch-10)//40))


def fit_with_scheduler(lamda, n_epochs, n_batch, scheduler):
    x_train, x_val, x_test, y_train, y_val, y_test, mean, std = load_data(os.path.join(
        os.getcwd(), 'Data', 'cifar-10-batches-py'))
    W = np.random.normal(0, 0.1, size=(n_out, n_in))
    b = np.random.normal(0, 0.1, (n_out, 1))
    logs = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    for i in range(n_epochs):
        e = scheduler(i)
        print(i, e)
        W, b, log = fit(x_train, y_train, W, b, lamda, e,
                        n_batch, 1, x_val, y_val)
        logs['train_loss'].extend(log['train_loss'])
        logs['val_loss'].extend(log['val_loss'])
        logs['train_acc'].extend(log['train_acc'])
        logs['val_acc'].extend(log['val_acc'])
    plt.plot([i+1 for i in range(n_epochs)],
             logs['train_loss'], label='trainig loss')
    plt.plot([i+1 for i in range(n_epochs)],
             logs['val_loss'], label='validation loss')
    plt.title(
        f'Loss function over training, lambda={lamda}')
    plt.legend()
    plt.savefig(f'results/scheduled_loss_over_trainig_lambda={lamda}.png')
    plt.close()

    plt.plot([i+1 for i in range(n_epochs)],
             logs['train_acc'], label='trainig accuracy')
    plt.plot([i+1 for i in range(n_epochs)],
             logs['val_acc'], label='validation accuracy')
    plt.title(f'Accuracy over training, lambda={lamda}')
    plt.legend()
    plt.savefig(f'results/scheduled_accuracy_over_trainig_lambda={lamda}.png')
    plt.close()

    # evaluation over test set
    eval = evaluate(x_test, y_test, W, b, lamda)
    test_loss = eval['loss']
    test_cost = eval['cost']
    test_acc = eval['acc']
    test_str = f'testing, lambda={lamda}, loss:{test_loss}, cost:{test_cost}, accuracy:{test_acc}'
    print(test_str)
    with open(f'results/scheduled_lambda={lamda}.txt', 'a') as f:
        f.write(test_str)


def plot_correct_hist(X, Y, W, b, is_mbce, save):
    Y_pred = forward(X, W, b, is_mbce)
    pred_index = Y_pred.argmax(axis=0)
    true_index = Y.argmax(axis=0)
    correct_samples = np.where(pred_index == true_index)[0]
    print(correct_samples.shape)
    print(np.sum(pred_index == true_index))
    print(np.array([Y_pred[pred_index[idx]][idx]
          for idx in correct_samples]).shape)
    plt.hist([Y_pred[pred_index[idx]][idx] for idx in correct_samples], bins=20,
             alpha=0.5, label='Correcty classified', color='red', range=[0, 1])
    plt.hist([Y_pred[pred_index[i]][i] for i in range(Y_pred.shape[1])], bins=20, alpha=0.5,
             label='All samples', color='blue', range=[0, 1])
    plt.legend()
    plt.title(
        'Cross_entropy with softmax' if not is_mbce else 'Multiple class binary classification')
    plt.savefig(save)
    plt.close()


def bonus_2_2():
    x_train, x_val, x_test, y_train, y_val, y_test, mean, std = load_data(os.path.join(
        os.getcwd(), 'Data', 'cifar-10-batches-py'))
    lamda = 0
    W_mbce = np.random.normal(0, 0.1, size=(n_out, n_in))
    b_mbce = np.random.normal(0, 0.1, (n_out, 1))
    W_mbce, b_mbce, logs = fit(x_train, y_train, W_mbce, b_mbce, lamda, 0.01,
                               100, 40, x_val, y_val, flipping=False, is_mbce=True)
    eval = evaluate(x_test, y_test, W_mbce, b_mbce, lamda, True)
    test_loss = eval['loss']
    test_cost = eval['cost']
    test_acc = eval['acc']
    test_str = f'testing mbce model: lambda={lamda}, accuracy:{test_acc}, loss:{test_loss}'
    print(test_str)
    with open(f'results/mbce_lambda={lamda}.txt', 'a') as f:
        f.write(test_str)
    plot_correct_hist(x_test, y_test, W_mbce, b_mbce, True,
                      f'results/mbce_lamb{lamda}.png')
    plt.plot(logs['val_loss'], label='validation loss')
    plt.plot(logs['train_loss'], label='trainig loss')
    plt.legend()
    plt.savefig('results/mbce_loss.png')
    plt.close()
    W = np.random.normal(0, 0.1, size=(n_out, n_in))
    b = np.random.normal(0, 0.1, (n_out, 1))
    W, b, logs = fit(x_train, y_train, W, b, lamda, 0.001,
                     100, 40, x_val, y_val, flipping=False, is_mbce=False)
    eval = evaluate(x_test, y_test, W, b, lamda, True)
    test_loss = eval['loss']
    test_cost = eval['cost']
    test_acc = eval['acc']
    test_str = f'testing simple model: lambda={lamda}, accuracy:{test_acc}, loss:{test_loss}'
    print(test_str)
    with open(f'results/mbce_lambda={lamda}.txt', 'a') as f:
        f.write(test_str)

    plot_correct_hist(x_test, y_test, W, b,
                      False, f'results/simple_lamda{lamda}.png')

    plt.plot(logs['val_loss'], label='validation loss')
    plt.plot(logs['train_loss'], label='trainig loss')
    plt.legend()
    plt.savefig('results/simple_loss.png')
    plt.close()


if __name__ == '__main__':
    # uncomment each of the functions to run that test
    # mandatory()

    # bonus_2_1_1()

    # bonus_2_1_2()
    # fit_with_scheduler(0.1, 100, 100, lr_scheduler)
    # bonus_2_2()
    pass
