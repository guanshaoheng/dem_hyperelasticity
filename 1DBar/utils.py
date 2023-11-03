
import torch
from torch.autograd import grad
import numpy as np
import time



dev = torch.device('cpu')
if torch.cuda.is_available():
    print("CUDA is available, running on GPU")
    dev = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print("CUDA not available, running on CPU")


# --------------------------------------------------------------------------------------------
#               MATERIAL CLASS
# --------------------------------------------------------------------------------------------
class MaterialModel:
    # ---------------------------------------------------------------------------------------
    # Purpose: Construction method
    # ---------------------------------------------------------------------------------------
    def __init__(self):
        print("Material setup !")

    def getEnergyBar1D(self, u, x):
        dudx = grad(u, x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]            # 计算位移对位置的一阶导数
        # 如果在第一步的创建网络时候不将网络的偏置设置为0，则在此处可能得到 dudx<-1，则计算会出现NAN
        energy = (1 + dudx) ** (3/2) - 3/2*dudx - 1                                                                   # 计算每个点的应变能密度
        return energy

    def getStrongForm(self, u, x):
        dudx = grad(u, x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]            # 计算位移对位置的一阶导数
        dWdE = 3/2 * ((1 + dudx)**0.5 - 1)                                                                            # 计算应变能对应变的导数，则为每个点的应力
        strong = grad(dWdE, x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0] + x   # 计算应力对未知的导数，再加上位移的坐标
        return strong
    


# -------------------------------------------------------------------------------
# Purpose: setting domain and collect database
# -------------------------------------------------------------------------------
def setup_domain(x_min, Length, Nx):
    # create points
    return np.linspace(x_min, Length, Nx)[:, np.newaxis]


# -----------------------------------------------------------------------------------------------------
# prepare inputs for testing the model
# -----------------------------------------------------------------------------------------------------
def get_datatest(x_min, Length, num_test_x):
    # create points
    return np.sort(np.random.uniform(x_min, Length, size=num_test_x))[:, np.newaxis]


# --------------------------------------------------------------------------
#           NEURAL NETWORK CLASS
# --------------------------------------------------------------------------
class MultiLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(MultiLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H, bias=True)
        # self.linear1_1 = torch.nn.Linear(H, H, bias=True)
        # self.linear1_2 = torch.nn.Linear(H, H, bias=True)
        self.linear2 = torch.nn.Linear(H, D_out, bias=True)

        # 为什么要把此处的偏置初始化为0, 如果此处不置为0，则在计算dudx的时候会小于-1，则计算能量密度的时候会出现NAN
        torch.nn.init.constant_(self.linear1.bias, 0.)
        # torch.nn.init.constant_(self.linear1_1.bias, 0.)
        # torch.nn.init.constant_(self.linear1_2.bias, 0.)
        torch.nn.init.constant_(self.linear2.bias, 0.)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        y1 = torch.tanh(self.linear1(x))
        # y1 = torch.tanh(self.linear1_1(y1))
        # y1 = torch.tanh(self.linear1_2(y1))
        y = self.linear2(y1)

        # # cal the dydx
        # dydx_grad = grad(y, x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        # dydx = torch.einsum('ij, jk->ki', self.linear2.weight, (1-torch.tanh(self.linear1(x))**2).T * self.linear1.weight)
        return y
    
    def dydx(self, x):
        y = self.linear2(torch.tanh(self.linear1(x)))

        dydx = grad(y, x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]

        return y, dydx


# --------------------------------------------------------------------------------
#       MAIN CLASS: Deep Energy Method
# --------------------------------------------------------------------------------
class DeepEnergyMethod:
    # Instance attributes
    def __init__(self, num_in, num_width, num_out, xmin, length):
        self.model = MultiLayerNet(num_in, num_width, num_out).to(dev)
        self.xmin = xmin; self.length = length

    # ------------------------------------------------------------------
    # Purpose: training model
    # 最关键的一步，根据最小能量法训练模型
    # ------------------------------------------------------------------
    def train_model(self, data, material_model, iteration, type_energy, integration, learning_rate):
        x = torch.from_numpy(data).float()
        x = x.to(dev)
        x.requires_grad_(True)
        optimizer = torch.optim.LBFGS(self.model.parameters(), lr=learning_rate, max_iter=20)
        start_time = time.time()
        energy_loss_array = []
        boundary_loss_array = []
        loss_array = []
        it_time = time.time()
        for t in range(iteration):
            # Zero gradients, perform a backward pass, and update the weights.
            def closure():
                # https://pytorch.org/docs/stable/optim.html, The closure should clear the gradients, compute the loss, and return it.
                # ----------------------------------------------------------------------------------
                # Internal Energy
                # ----------------------------------------------------------------------------------
                u_pred = self.getU(x)     # 预测满足位移边界条件 (x+1==0, 位移为0) 的位移
                u_pred.double()
                # Strain energy equations = Internal Energy
                if type_energy == "Bar1D":
                    potential_energy = material_model.getEnergyBar1D(u_pred, x)  # 计算每个点上的应变能密度
                else:
                    print("Error: Please specify type model !!!")
                    exit()
                
                """
                    f 这一项至关重要

                    要是不添加这一项, 在优化中, 位移直接为0, 则potential也为0即可得到满足优化目标的解
                    因此, 添加该项是为了最终得到的位移不为常数0
                """
                work_outer = x*u_pred
                if integration == 'montecarlo':
                    dom_crit = (self.length-self.xmin) * self.loss_sum(potential_energy) - (self.length-self.xmin) * self.loss_sum(work_outer)  # 第一项为能量积分 l*sum(w_density)，第二项为 -l*sum(u*x)
                elif integration == 'trapezoidal':
                    dom_crit = self.trapz1D(potential_energy, x=x) - self.trapz1D(work_outer, x=x)
                else:   # simpson积分
                    dom_crit = self.simps1D(potential_energy, x=x) - self.simps1D(work_outer, x=x)
                # ----------------------------------------------------------------------------------
                # Compute and print loss
                # ----------------------------------------------------------------------------------
                energy_loss = dom_crit
                boundary_loss = torch.tensor([0])
                loss = energy_loss
                optimizer.zero_grad()
                loss.backward()
                print('Iter: %d Loss: %.9e Energy: %.9e Boundary: %.9e Time: %.3e'
                      % (t + 1, loss.item(), energy_loss.item(), boundary_loss.item(), time.time() - it_time))
                energy_loss_array.append(energy_loss.data)
                boundary_loss_array.append(boundary_loss.data)
                loss_array.append(loss.data)
                return loss
            optimizer.step(closure)
        elapsedDEM = time.time() - start_time
        print('Training time: %.4f' % elapsedDEM)

    def getU(self, x):
        """
            输入点的位置，预测该点的位移

            注意: (x + 1) * self.model(x)  来确保输出位移满足位移边界条件(x+1=0 处, 位移为0)
        """
        return (x + 1) * self.model(x) 

    def simps(self, y, x=None, dx=1, axis=-1, even='avg'):
        nd = len(y.shape)
        N = y.shape[axis]
        last_dx = dx
        first_dx = dx
        returnshape = 0
        if x is not None:
            if len(x.shape) == 1:
                shapex = [1] * nd
                shapex[axis] = x.shape[0]
                saveshape = x.shape
                returnshape = 1
                x = x.reshape(tuple(shapex))
            elif len(x.shape) != len(y.shape):
                raise ValueError("If given, shape of x must be 1-d or the "
                                 "same as y.")
            if x.shape[axis] != N:
                raise ValueError("If given, length of x along axis must be the "
                                 "same as y.")
        if N % 2 == 0:
            val = 0.0
            result = 0.0
            slice1 = (slice(None),) * nd
            slice2 = (slice(None),) * nd
            if even not in ['avg', 'last', 'first']:
                raise ValueError("Parameter 'even' must be "
                                 "'avg', 'last', or 'first'.")
            # Compute using Simpson's rule on first intervals
            if even in ['avg', 'first']:
                slice1 = self.tupleset(slice1, axis, -1)
                slice2 = self.tupleset(slice2, axis, -2)
                if x is not None:
                    last_dx = x[slice1] - x[slice2]
                val += 0.5 * last_dx * (y[slice1] + y[slice2])  # 最后一个区间使用梯形公式
                result += self._basic_simps(y, 0, N - 3, x, dx, axis)
            # Compute using Simpson's rule on last set of intervals
            if even in ['avg', 'last']:
                slice1 = self.tupleset(slice1, axis, 0)
                slice2 = self.tupleset(slice2, axis, 1)
                if x is not None:
                    first_dx = x[tuple(slice2)] - x[tuple(slice1)]
                val += 0.5 * first_dx * (y[slice2] + y[slice1]) # 第一个区间使用梯形公式
                result += self._basic_simps(y, 1, N - 2, x, dx, axis)
            if even == 'avg':
                val /= 2.0
                result /= 2.0
            result = result + val
        else:
            result = self._basic_simps(y, 0, N - 2, x, dx, axis)
        if returnshape:
            x = x.reshape(saveshape)
        return result

    def tupleset(self, t: tuple, i: int, value: slice)->tuple:
        """
            Change the i-th element in the tuple to value 
        """
        l = list(t)
        l[i] = value
        return tuple(l)

    def _basic_simps(self, y, start, stop, x, dx, axis):
        nd = len(y.shape)
        if start is None:
            start = 0
        step = 2
        slice_all = (slice(None),) * nd
        slice0 = self.tupleset(slice_all, axis, slice(start, stop, step))
        slice1 = self.tupleset(slice_all, axis, slice(start + 1, stop + 1, step))
        slice2 = self.tupleset(slice_all, axis, slice(start + 2, stop + 2, step))

        if x is None:  # Even spaced Simpson's rule.
            result = torch.sum(dx / 3.0 * (y[slice0] + 4 * y[slice1] + y[slice2]), axis)
        else:
            # Account for possibly different spacings.
            #    Simpson's rule changes a bit.
            # h = np.diff(x, axis=axis)
            h = self.torch_diff_axis_0(x, axis=axis)
            sl0 = self.tupleset(slice_all, axis, slice(start, stop, step))
            sl1 = self.tupleset(slice_all, axis, slice(start + 1, stop + 1, step))
            h0 = h[sl0]
            h1 = h[sl1]
            hsum = h0 + h1
            hprod = h0 * h1
            h0divh1 = h0 / h1
            tmp = hsum / 6.0 * (y[slice0] * (2 - 1.0 / h0divh1) +
                                y[slice1] * hsum * hsum / hprod +
                                y[slice2] * (2 - h0divh1))
            result = torch.sum(tmp, dim=axis)
        return result

    def torch_diff_axis_0(self, a, axis):
        if axis == 0:
            return a[1:, 0:1] - a[:-1, 0:1]
        elif axis == -1:
            return a[1:] - a[:-1]
        else:
            print("Not implemented yet !!! function: torch_diff_axis_0 error !!!")
            exit()

    def simps1D(self, f, x=None, dx=1.0, axis=-1):
        f1D = f.flatten()
        if x is not None:
            x1D = x.flatten()
            return self.simps(f1D, x1D, dx=dx, axis=axis)
        else:
            return self.simps(f1D, dx=dx, axis=axis)

    def __trapz(self, y, x=None, dx=1.0, axis=-1):
        # y = np.asanyarray(y)
        """
        # 原 代码
        if x is None:
            d = dx
        else:
            d = x[1:] - x[0:-1]
            # reshape to correct shape
            shape = [1] * y.ndimension()
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        nd = y.ndimension()
        slice1 = [slice(None)] * nd
        slice2 = [slice(None)] * nd
        slice1[axis] = slice(1, None)
        slice2[axis] = slice(None, -1)
        ret = torch.sum(d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0, axis)
        return ret
        """
        if x is None:
            d = dx
        else:
            d = x[1:] - x[:-1]
        return torch.sum(d * (y[:-1] + y[1:])*0.5)

    def trapz1D(self, y, x=None, dx=1.0, axis=-1):
        y1D = y.flatten()
        if x is not None:
            x1D = x.flatten()
            return self.__trapz(y1D, x1D, dx=dx, axis=axis)
        else:
            return self.__trapz(y1D, dx=dx)

    def get_gradu(self, u, x):
        dudx = grad(u, x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        return dudx

    def evaluate_model(self, x_space):
        t_tensor = torch.from_numpy(x_space).float()
        t_tensor = t_tensor.to(dev)
        t_tensor.requires_grad_(True)
        u_pred_torch = self.getU(t_tensor)
        u_pred = u_pred_torch.detach().cpu().numpy()
        dudx_torch = grad(u_pred_torch, t_tensor, torch.ones(t_tensor.shape[0], 1, device=dev))[0]
        dudx = dudx_torch.detach().cpu().numpy()
        return u_pred, dudx

    # --------------------------------------------------------------------------------
    # Purpose: loss sum for the energy part
    # --------------------------------------------------------------------------------
    @staticmethod
    def loss_sum(tinput):
        return torch.sum(tinput) / tinput.data.nelement()

    # --------------------------------------------------------------------------------
    # purpose: loss square sum for the boundary part
    # --------------------------------------------------------------------------------
    @staticmethod
    def loss_squared_sum(tinput, target):
        row, column = tinput.shape
        loss = 0
        for j in range(column):
            loss += torch.sum((tinput[:, j] - target[:, j]) ** 2) / tinput[:, j].data.nelement()
        return loss


# --------------------------------------------------------------------------------
# purpose: doing something in post processing for visualization
# --------------------------------------------------------------------------------
def write_vtk(filename, x_space, y_space, z_space, Ux, Uy, Uz):
    xx, yy, zz = np.meshgrid(x_space, y_space, z_space)
    displacement = (Ux, Uy, Uz)
    gridToVTK(filename, xx, yy, zz, pointData={"displacement": displacement})

