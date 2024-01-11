from import_file import *
import config as cf 
from Utility import cal_jacobian



def kernel(X1, X2, sigma=1.0):
    """
        X inshape of (nums_points, 2 or 3)
    """
    k = torch.exp(-torch.norm((X1.unsqueeze(1) - X2), dim=2)**2 /(2. * sigma**2))
    return k


class GP_Lagrange:
    def __init__(self, E=cf.E, nu=cf.nu, dim=2) -> None:
        self.nugget = 1e-5
        self.dim =dim
        # 材料参数
        self.mu = E / (2 * (1 + nu))
        self.lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))

        XX = np.meshgrid(np.linspace(0., cf.lx, cf.nx), np.linspace(0., cf.ly, cf.ny))
        X = np.concatenate((XX[0].reshape(-1, 1), XX[1].reshape(-1, 1)), axis=1)
        
        self.X = torch.from_numpy(X).float() # plt.scatter(X[:, 0], X[:, 1])
        self.XX = self.X.reshape(cf.nx, cf.ny, self.dim).permute(2, 0, 1)

        self.K = kernel(self.X, self.X) + torch.eye(cf.nx*cf.ny) * self.nugget 
        self.K_inv = torch.linalg.inv(self.K)

        U = np.zeros([cf.nx, cf.ny, self.dim])

        # 施加边界条件
        left_inx = np.where(self.XX[0] == 0.)
        right_inx = np.where(self.XX[0] == cf.lx)
        U[left_inx[0], left_inx[1], 0] = 0.    # 左边界为 0
        U[left_inx[0], left_inx[1], 1] = 0.    
        U[right_inx[0], right_inx[1], 0] = 0.1 * cf.lx  # 右边界为0.1
        mask = torch.zeros(U.shape, dtype=torch.bool)  # 需要采用hook将其梯度设置为0的点
        mask[left_inx[0],  left_inx[1],  0] = True
        mask[left_inx[0],  left_inx[1],  1] = True
        mask[right_inx[0], right_inx[1], 0] = True
        self.U = torch.nn.Parameter(torch.from_numpy(U).float(), requires_grad=True)
        # self.U[1:-1, :, :].requires_grad = True
        # self.U[-1, :, 1].requires_grad = True

        def gradient_hook(grad):
            grad[mask] = 0.
            return grad
        
        self.U.register_hook(gradient_hook)


        # 测试点
        XX_ = torch.meshgrid(torch.linspace(0., cf.lx, cf.nx_test), torch.linspace(0., cf.ly, cf.ny_test))  
        self.X_ = torch.stack([XX_[i].reshape(-1) for i in range(len(XX_))]).permute(1, 0)
        self.X_.requires_grad = True
        self.K_ = kernel(self.X_, self.X)
        self.prefix = self.K_ @ self.K_inv

        # 定义优化器
        self.optimizer = torch.optim.Adam([self.U], lr=0.001*0.1)

    def get_u(self,):
        U_ =  self.prefix @ self.U.reshape(-1, 2)
        return U_
    
    def get_dudX(self, U_):
        tmp = torch.stack(
                [
                    torch.autograd.grad(
                        U_[:, i].unsqueeze(1),
                        self.X_,
                        grad_outputs=torch.ones(size=[len(U_), 1]),
                        retain_graph=True,
                        create_graph=True,
                    )[0]
                    for i in range(U_.size(1))
                ],
                dim=-1,
            ).permute(0, 2, 1)
        return tmp
    
    def strain_energy_NeoHookean2D(self, dudX):
        """
            势能方程参考：https://en.wikipedia.org/wiki/Neo-Hookean_solid
        """
        F = dudX + torch.eye(2)
        C = torch.einsum("nji, njk->nik", F, F)
        J = torch.det(F)
        I_1 = torch.einsum("nii->n", C)
        strainEnergy = 0.5 * self.mu * (I_1 - 2 - 2*torch.log(J)) + 0.5*self.lam * (J-1)**2

        # 计算PK1应力
        P = torch.autograd.grad(
            strainEnergy, 
            F, 
            grad_outputs= torch.ones_like(strainEnergy),
            retain_graph=True,
            create_graph=True)[0]
        # 计算柯西应力
        S = torch.einsum("nij, njk->nik", torch.linalg.inv(F), P) 
        return strainEnergy, S
    
    def train(self,):
        for t in range(cf.iteration):
            it_time = time.time()
            u_pre = self.get_u()
            dudX = self.get_dudX(u_pre)
            energy_density, S = self.strain_energy_NeoHookean2D(dudX=dudX)

            energy_density = energy_density.reshape(cf.nx_test, cf.ny_test)

            internal = 0.25 * (
                2.*(torch.sum(energy_density[1:-1, :]) + torch.sum(energy_density[:, 1:-1]))+
                energy_density[0, 0]  + energy_density[0, -1] + 
                energy_density[-1, 0]  + energy_density[-1, -1] ) * cf.hx_test * cf.hy_test
            S_boundary = S.reshape(cf.nx_test, cf.ny_test, self.dim, self.dim)[-1]  # 力边界上的PK2
            t_boundary = torch.einsum("nij, j-> ni", S_boundary, torch.tensor([1.0, 0.], dtype=torch.float32))
            external = -torch.sum(0.5*(t_boundary[1:, 0] + t_boundary[:-1, 0])*cf.hx_test)
            loss = internal  + external
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            line = 'Iter: %d Loss: %.9e Internal: %.9e External: %.9e Time: %.3e(s)' \
                    % (t + 1, loss.item(), internal.item(), external.item(),
                       (time.time() - it_time))
            print(line)
        
        self.plot(self.U[:, :, 0])
        self.plot(self.U[:, :, 1])

    def plot(self, A):
        """

        """
        # plt.close()
        plt.contourf(
            self.XX[0].detach().numpy(), 
            self.XX[1].detach().numpy(), 
            A.detach().numpy(),
            # self.U.detach().numpy()[:, :, 0], 
            cmap = 'RdBu')
        # plt.imshow(A.detach().numpy())
        plt.colorbar()
        plt.show()
        
    def write_vtk(self, filename="test"):
        tmp = self.XX.detach().numpy().astype(np.float64)
        xx = np.stack([tmp[0], tmp[0]]).transpose(1, 2, 0)
        yy = np.stack([tmp[1], tmp[1]]).transpose(1, 2, 0)
        zz = np.zeros_like(xx)
        zz[:, :, 1] = 1e-6
        u = self.U.detach().numpy().astype(np.float64)
        u = u[:, :, np.newaxis, :]
        u = np.concatenate((u, u), axis=2)

        """
             C-contiguous 的时候，它的元素在内存中是按行优先顺序存储的。
             也就是说，在二维数组中，第一行的所有元素在内存中是连续存放的，
             接着是第二行的所有元素，以此类推.
             由于处理器缓存的方式，访问连续内存位置通常更快。

             如果一个数组不是 C-contiguous，可以通过使用 np.ascontiguousarray 
             函数来创建一个 C-contiguous 的数组副本
        """
        
        xx_ = np.ascontiguousarray(xx)
        yy_ = np.ascontiguousarray(yy)
        zz_ = np.ascontiguousarray(zz)

        gridToVTK(filename + "_original", 
                    xx_,
                    yy_,
                    zz_, 
                    pointData={
                        "ux": np.ascontiguousarray(u[..., 0]), 
                        "uy": np.ascontiguousarray(u[..., 1])})
        
        xx_ = np.ascontiguousarray(xx + u[..., 0])
        yy_ = np.ascontiguousarray(yy + u[..., 1])

        saved_name = filename + "_deformed"
        gridToVTK(saved_name, 
                    xx_,
                    yy_,
                    zz_, 
                    pointData={
                        "ux": np.ascontiguousarray(u[..., 0]), 
                        "uy": np.ascontiguousarray(u[..., 1])})
        print("="*60)
        print(f"VTK file saved as {saved_name:s}")
        print("="*60)
        

def main():
    obj = GP_Lagrange()
    obj.train()
    obj.write_vtk()

    # 输出结果到vts




if __name__ == "__main__":
    main()