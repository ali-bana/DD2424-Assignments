from model import MLP
from utils import get_data, get_all_data
import os
import matplotlib.pyplot as plt
import numpy as np
import itertools
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
