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


if __name__ == '__main__':
    x_train, x_val, x_test, y_train, y_val, y_test, mean, std = get_data(
        os.path.join(os.getcwd(), 'Data'))

    print(x_train.shape, x_val.shape, x_test.shape)
    print(y_train.shape, y_val.shape, y_test.shape)
    print(x_train.mean(), x_train.std())
    print(x_test.mean(), x_test.std())

    print(x_val.mean(), x_val.std())
