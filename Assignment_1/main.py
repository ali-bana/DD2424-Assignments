from cProfile import label
from utils import load_data, display_flat_image
from functions import ComputeGradsNum, ComputeGradsNumSlow, montage
from simpler import forward, compute_gradients, fit, evaluate
import numpy as np
import os
import matplotlib.pyplot as plt

n_in = 3072
n_out = 10


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


def fit_and_plot(x_train, x_val, x_test, y_train, y_val, y_test, lamda, n_epochs, n_batch, eta):
    W = np.random.normal(0, 0.1, size=(n_out, n_in))
    b = np.random.normal(0, 0.1, (n_out, 1))
    W, b, logs = fit(x_train, y_train, W, b, lamda, eta,
                     n_batch, n_epochs, x_val, y_val)
    plt.plot([i+1 for i in range(n_epochs)],
             logs['train_loss'], label='trainig loss')
    plt.plot([i+1 for i in range(n_epochs)],
             logs['val_loss'], label='validation loss')
    plt.title(
        f'Loss function over training eta={eta}, lambda={lamda}')
    plt.legend()
    plt.savefig(f'results/loss_over_trainig_eta={eta}_lambda={lamda}.png')
    plt.show()

    plt.plot([i+1 for i in range(n_epochs)],
             logs['train_cost'], label='trainig cost')
    plt.plot([i+1 for i in range(n_epochs)],
             logs['val_cost'], label='validation cost')
    plt.title(
        f'Cost function over training eta={eta}, lambda={lamda}')
    plt.legend()
    plt.savefig(f'results/cost_over_trainig_eta={eta}_lambda={lamda}.png')
    plt.show()

    plt.plot([i+1 for i in range(n_epochs)],
             logs['train_acc'], label='trainig accuracy')
    plt.plot([i+1 for i in range(n_epochs)],
             logs['val_acc'], label='validation accuracy')
    plt.title(f'Accuracy over training eta={eta}, lambda={lamda}')
    plt.legend()
    plt.savefig(f'results/accuracy_over_trainig_eta={eta}_lambda={lamda}.png')
    plt.show()

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


if __name__ == '__main__':
    x_train, x_val, x_test, y_train, y_val, y_test, mean, std = load_data(os.path.join(
        os.getcwd(), 'Data', 'cifar-10-batches-py'))

    # 20 samples is enough and does not slow down the calculations
    gradient_checker(x_train[:, :20], y_train[:, :20], 0.1, True)
    fit_and_plot(x_train, x_val, x_test, y_train,
                 y_val, y_test, 0, 40, 100, 0.1)

    fit_and_plot(x_train, x_val, x_test, y_train,
                 y_val, y_test, 0, 40, 100, 0.001)

    fit_and_plot(x_train, x_val, x_test, y_train,
                 y_val, y_test, 0.1, 40, 100, 0.001)

    fit_and_plot(x_train, x_val, x_test, y_train,
                 y_val, y_test, 1, 40, 100, 0.001)
