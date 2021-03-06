from cProfile import label

from scipy.fft import idct
from utils import load_data, display_flat_image
from functions import ComputeGradsNum, ComputeGradsNumSlow, montage
from simpler import forward, compute_gradients, fit, evaluate
import numpy as np
import os
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm

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
    # mandatory()

    # bonus_2_1_1()

    # bonus_2_1_2()
    # fit_with_scheduler(0.1, 100, 100, lr_scheduler)
    bonus_2_2()
