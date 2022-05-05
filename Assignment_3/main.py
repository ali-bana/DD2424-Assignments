import numpy as np
import matplotlib.pyplot as plt
import os
from model import Model
from utils import get_all_data, flip


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
