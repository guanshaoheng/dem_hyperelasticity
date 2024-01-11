import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import jv

def data_gen(n=20, save_path='xy_data', gaussian_flag=True, noise=0.1, k=4.0):
    x, X, Y = input_2d(num=n)
    if gaussian_flag:
        kernel = kernel_bessel_2d(x=x, k=k)
        y = np.random.multivariate_normal(mean=np.zeros(n**2), cov=kernel)
    else:
        omega = 2.0
        phi = 0.2
        A = 1.0
        y = A * np.sin(omega * (x[:, 0] + x[:, 1]) + phi)

    # save the data
    y_noise = y + np.random.normal(0, 1, len(y))*noise

    # plot the data Truth
    fig = plt.figure()
    plt.contourf(X, Y, y.reshape(n, n), cmap = 'RdBu',
                             vmin = -3.0,
                             vmax = 2.4)
    plt.colorbar()
    plt.tight_layout()
    fig_path_name = os.path.join(save_path, 'contourf_helmholtz_Truth.png')
    plt.savefig(fig_path_name, dpi=200)
    plt.close()

    # plot the data noise
    num_used_train = int(len(index)*0.2)
    index_train = index[:num_used_train]
    index_test = index[-num_used_train:]
    plt.contourf(X, Y, y_noise.reshape(n, n), cmap = 'RdBu',
                             vmin = -3.0,
                             vmax = 2.4)
    plt.colorbar()
    plt.scatter(x[index_train, 0], x[index_train, 1], label='Training sets', c='g', marker='x')
    plt.scatter(x[index_test, 0], x[index_test, 1], label='Validation sets', c='k', marker='o')
    plt.legend()
    plt.tight_layout()
    fig_path_name = os.path.join(save_path, 'contourf_helmholtz_noise.png')
    fig = plt.gcf()
    fig.savefig(fig_path_name, dpi=200)
    # plt.show()
    plt.close()
    return

def bessel_func(v, x):
    '''

    :param v:  number of the alpha
    :param x:  vector of the input
    :return:
    '''
    return jv(v, x)

def kernel_bessel_2d(x, k):
    r'''
     this kernel func is referred to the paper
         [1] C. Albert, “Physics-informed transfer path analysis with parameter
             estimation using Gaussian processes,” Proc. Int. Congr. Acoust., vol. 2019-Septe, no. 1, pp. 459–466, 2019, doi: 10.18154/RWTH-CONV-238988.
    :param x:  in shape of (num_samples, (input_1, input_2))
    :param k:  $J_0( k \| x-x' \| )$
    :return:
    '''
    distance = cal_distance(x)
    kernel = bessel_func(v=0, x=k*distance)
    # plt.imshow(kernel); plt.colorbar();plt.show()
    return kernel


def cal_distance(x):
    '''
        calculate the distance of the input
    :param x: in shape of (num_samples, (input_1, input_2))
    :return:
    '''
    n = len(x)
    distance = np.zeros(shape=[n, n])
    for i in range(n):
        for j in range(n):
            distance[i, j] = np.linalg.norm(x[i] - x[j])
    return distance


def input_2d(min_=-1, max_=1, num=11):
    x = np.linspace(min_, max_, num)
    X = np.meshgrid(x, x)
    xy = np.concatenate((X[0].reshape(-1, 1), X[1].reshape(-1, 1)), axis=1)
    return xy, X[0], X[1]


def main():
    x, X, Y = input_2d()
    kernel = kernel_bessel_2d(x=x, k=2.)
    plt.imshow(kernel); plt.colorbar(); plt.show()


if __name__ == '__main__':
    data_gen()