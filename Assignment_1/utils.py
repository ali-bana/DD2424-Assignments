import matplotlib.pyplot as plt
import numpy as np
import os


def unpickle(file):
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

    # img = img * std + mean
    # # print(img.max(), img.mean(), img2.max(), img2.min())
    # reshaped = img.reshape([3, 32, 32])
    # r = reshaped[0, :, :]
    # g = reshaped[1, :, :]
    # b = reshaped[2, :, :]
    # plt.imshow(np.dstack((r, g, b))/255)
    # plt.show()


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


if __name__ == '__main__':
    x_train, x_val, x_test, y_train, y_val, y_test, mean, std = load_data(os.path.join(
        os.getcwd(), 'Data', 'cifar-10-batches-py'))

    # for i in range(32 * 3 - 1):
    #     x_train[:, 0][i*32:(i+1)*32] = x_train[:, 0][i*32:(i+1)*32][::-1]
    x_train_f = flip(x_train)
    for i in range(10):
        display_flat_image(x_train[:, i])
        display_flat_image(x_train_f[:, i])
