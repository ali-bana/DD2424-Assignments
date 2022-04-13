from model import MLP
from utils import get_data
import os
import matplotlib.pyplot as plt
input_size = 3072


def make_circular_scheduler(max_lr, min_lr, n_s):
    def scheduler(iter, lr):
        quotient = iter // n_s
        if quotient % 2 == 0:  # increasing part
            return min_lr + (iter - quotient*n_s) * (max_lr - min_lr) / n_s
        else:  # decreasing part
            return max_lr - (iter - quotient*n_s) * (max_lr - min_lr) / n_s
    return scheduler


def logs_plotter(logs, name):
    n_epochs = len(logs['train_loss'])
    plt.plot([i+1 for i in range(n_epochs)],
             logs['train_loss'], label='train loss')
    plt.plot([i+1 for i in range(n_epochs)],
             logs['val_loss'], label='validation loss')
    plt.legend()
    plt.title('Loss over training')
    plt.savefig(f'results/{name}_loss.png')
    plt.close()

    plt.plot([i+1 for i in range(n_epochs)],
             logs['train_cost'], label='train cost')
    plt.plot([i+1 for i in range(n_epochs)],
             logs['val_cost'], label='validation cost')
    plt.legend()
    plt.title('Cost over training')
    plt.savefig(f'results/{name}_cost.png')
    plt.close()

    plt.plot([i+1 for i in range(n_epochs)],
             logs['train_acc'], label='train accuracy')
    plt.plot([i+1 for i in range(n_epochs)],
             logs['val_acc'], label='validation accuracy')
    plt.legend()
    plt.title('Accuracy over training')
    plt.savefig(f'results/{name}_acc.png')
    plt.close()


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
    # plt.plot([scheduler(0, i) for i in range(100)])
    # plt.show()
    # =========================================================
    # Exercise 3 train with cyclic
    scheduler = make_circular_scheduler(0.1, 1e-5, 500)
    model = MLP(input_size, 50, 10, 0.01)
    logs = model.fit(x_train, y_train, 0.01, 10, 100, x_val, y_val, scheduler)
    logs_plotter(logs, 'temp')
    # =========================================================


if __name__ == '__main__':
    main()
