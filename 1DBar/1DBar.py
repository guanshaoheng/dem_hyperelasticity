import torch
from torch.autograd import grad
import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt
import matplotlib as mpl
import time
from pyevtk.hl import gridToVTK

from utils import MaterialModel, get_datatest, setup_domain, DeepEnergyMethod
# import scipy as sp
# from graphviz import Digraph
# from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D
# from torch.autograd import Variable
# make_dot was moved to https://github.com/szagoruyko/pytorchviz
# from torchviz import make_dot

dev = torch.device('cpu')
if torch.cuda.is_available():
    print("CUDA is available, running on GPU")
    dev = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print("CUDA not available, running on CPU")
mpl.rcParams['figure.dpi'] = 100
# fix random seeds
axes = {'labelsize' : 'large'}
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 17}
legend = {'fontsize': 'medium'}
lines = {'linewidth': 3,
         'markersize' : 7}
mpl.rc('font', **font)
mpl.rc('axes', **axes)
mpl.rc('legend', **legend)
mpl.rc('lines', **lines)
npr.seed(2019)
torch.manual_seed(2019)

# ANALYTICAL SOLUTION
# exact = lambda X: 1. / 135. * X * (3 * X ** 4 - 40 * X ** 2 + 105)
# 设置解析解，位移是
exactU = lambda X: 1/135 * (68 + 105*X - 40*X**3 + 3*X**5)  
exactStrain = lambda x: 1./9. * (x**4 - 8*x**2 + 7)
exactEnergy = lambda eps: (1+eps)**(3/2) - 3/2*eps - 1

# ------------------------------ network settings ---------------------------------------------------
iteration = 30  
D_in = 1                           # 网络输入参数个数
H = 10                             # 网络隐含层宽度
D_out = 1                          # 输出参数个数
learning_rate = 1.0                # 学习率

# ------------------------------ material parameter -------------------------------------------------
model_energy = 'Bar1D'             # simu name  

# ----------------------------- define structural parameters ---------------------------------------
Length = 1.0                       # 长度
known_left_ux = 0                  # 
bc_left_penalty = 1.0              #
        
known_right_tx = 0                 # 
bc_right_penalty = 1.0             #
# ------------------------------ define domain and collocation points -------------------------------
Nx = 1000                          # 全局长度上的所有的点的个数， 训练点的个数
x_min = -1                         # 长度上的起点
h = (Length - x_min) / (Nx-1)      # 间隔
# ------------------------------ data testing -------------------------------------------------------
num_test_x = 100                   # 测试点的个数
# ------------------------------ filename output ----------------------------------------------------
# ------------------------------ filename output ----------------------------------------------------


# ----------------------------------------------------------------------
#                   EXECUTE PROGRAMME
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # ----------------------------------------------------------------------
    #                   STEP 1: SETUP DOMAIN - COLLECT CLEAN DATABASE
    # ----------------------------------------------------------------------
    dom = setup_domain(x_min=x_min, Length=Length, Nx=Nx)                           # 设置训练区域
    x_predict = get_datatest(x_min=x_min, Length=Length, num_test_x=num_test_x)     # 设置测试点的区域
    exact_solution = exactU(x_predict)                                              # 计算精确解
    exact_eps = exactStrain(x_predict)                                              # 计算精确解对应的应变值
    # ----------------------------------------------------------------------
    #                   STEP 2.1: SETUP DEM MODEL
    # ----------------------------------------------------------------------
    mat = MaterialModel()    # 创建材料模型

    # 优化模型中采用 simpson 积分
    dems = DeepEnergyMethod(D_in, H, D_out, xmin=x_min, length=Length)
    time_dems = time.time()
    dems.train_model(dom, mat, iteration, model_energy, 'simpson', learning_rate)
    time_dems = time.time() - time_dems
    u_pred_dems, eps_pred_dems = dems.evaluate_model(x_predict)
    error_L2_DEMS = np.linalg.norm(exact_solution - u_pred_dems, 2) / np.linalg.norm(exact_solution, 2)
    error_H1_DEMS = np.linalg.norm(exact_eps - eps_pred_dems, 2) / np.linalg.norm(exact_eps, 2)
    
    # 优化模型中采用 trapezoidal 积分  梯形公式积分
    demt = DeepEnergyMethod(D_in, H, D_out, xmin=x_min, length=Length)
    time_demt = time.time()
    demt.train_model(dom, mat, iteration, model_energy, 'trapezoidal', learning_rate)
    time_demt = time.time() - time_demt
    u_pred_demt, eps_pred_demt = demt.evaluate_model(x_predict)
    error_L2_DEMT = np.linalg.norm(exact_solution - u_pred_demt, 2) / np.linalg.norm(exact_solution, 2)
    error_H1_DEMT = np.linalg.norm(exact_eps - eps_pred_demt, 2) / np.linalg.norm(exact_eps, 2)

    # 优化模型中采用 montecarlo 积分  
    demm = DeepEnergyMethod(D_in, H, D_out, xmin=x_min, length=Length)
    time_demm = time.time()
    demm.train_model(dom, mat, iteration, model_energy, 'montecarlo', learning_rate)
    time_demm = time.time() - time_demm
    u_pred_demm, eps_pred_demm = demm.evaluate_model(x_predict)
    error_L2_DEMM = np.linalg.norm(exact_solution - u_pred_demm, 2) / np.linalg.norm(exact_solution, 2)
    error_H1_DEMM = np.linalg.norm(exact_eps - eps_pred_demm, 2) / np.linalg.norm(exact_eps, 2)

    
    # ----------------------------------------------------------------------
    #                                  PLOT
    # ----------------------------------------------------------------------
    fig , ax = plt.subplots(figsize=(10, 8))
    fig2, bx = plt.subplots(figsize=(10, 8))
    fig3, cx = plt.subplots(figsize=(10, 8))
    fig4, dx = plt.subplots(figsize=(10, 8))

    ax.plot(x_predict, exact_solution, label="Exact", linestyle='dashed', color='black')
    ax.plot(x_predict, u_pred_dems, 'rx', label='DEMS')
    ax.plot(x_predict, u_pred_demt, 'g+', label='DEMT')
    ax.plot(x_predict, u_pred_demm, 'b2', label='DEMM')
    ax.legend(ncol=4, fancybox=True, shadow=True, loc='upper left', bbox_to_anchor=(0.0, 1.14))
    ax.grid(True)
    ax.set_xlabel('X')
    ax.set_ylabel('u(X)')
    bx.plot(x_predict, exactStrain(x_predict), label='Exact', linestyle='dashed', color='black')
    bx.plot(x_predict, eps_pred_dems, 'rx', label='DEMS')
    bx.plot(x_predict, eps_pred_demt, 'g+', label='DEMT')
    bx.plot(x_predict, eps_pred_demm, 'b2', label='DEMM')
    legend = bx.legend(loc='upper left', shadow=True)
    bx.legend(ncol=4, fancybox=True, shadow=True, loc='upper left', bbox_to_anchor=(0.0, 1.14))
    bx.grid(True)
    bx.set_xlabel('X')
    bx.set_ylabel('du(X)/dX')
    cx.semilogy(x_predict, np.abs(exact_solution - u_pred_dems), 'r-.', label=r"$|u_{Exact} - u_{{DEMS}}|$")
    cx.semilogy(x_predict, np.abs(exact_solution - u_pred_demt), 'g-.', label='$|u_{Exact} - u_{DEMT}|$')
    cx.semilogy(x_predict, np.abs(exact_solution - u_pred_demm), 'b-.', label='$|u_{Exact} - u_{DEMM}|$')
    cx.grid(True)
    cx.set_xlabel('X')
    cx.set_ylabel('error')
    cx.legend(ncol=3, fancybox=True, shadow=True, loc='upper left', bbox_to_anchor=(-0.03, 1.14))
    dx.semilogy(x_predict, np.abs(exact_eps - eps_pred_dems), 'r-.', label='$|\epsilon_{Exact} - \epsilon_{DEMS}|$')
    dx.semilogy(x_predict, np.abs(exact_eps - eps_pred_demt), 'g-.', label='$|\epsilon_{Exact} - \epsilon_{DEMT}|$')
    dx.semilogy(x_predict, np.abs(exact_eps - eps_pred_demm), 'b-.', label='$|\epsilon_{Exact} - \epsilon_{DEMM}|$')
    dx.grid(True)
    dx.set_xlabel('X')
    dx.set_ylabel('error')
    dx.legend(ncol=3, fancybox=True, shadow=True, loc='upper left', bbox_to_anchor=(-0.03, 1.14))
    print("DEMS ||e||L2 : %.2e" % error_L2_DEMS)
    print("DEMT ||e||L2 : %.2e" % error_L2_DEMT)
    print("DEMM ||e||L2 : %.2e" % error_L2_DEMM)
    print("DEMS ||e||H1 : %.2e" % error_H1_DEMS)
    print("DEMT ||e||H1 : %.2e" % error_H1_DEMT)
    print("DEMM ||e||H1 : %.2e" % error_H1_DEMM)
    print("DEMS time : %.2f" % time_dems)
    print("DEMT time : %.2f" % time_demt)
    print("DEMM time : %.2f" % time_demm)
    plt.show()
