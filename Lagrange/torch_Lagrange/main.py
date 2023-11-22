# from import_file import *
import config as cf 
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from pyevtk.hl import gridToVTK


def ij_2_index(i, j):
    """
        每行有 ny 个点
    """
    return i*cf.ny + j

def cal_von_miese_stress(sig_xx, sig_xy, sig_yy):
    return np.sqrt(sig_xx**2 - sig_xx*sig_yy + sig_yy**2 + 3.* sig_xy**2)


class Torch_Lagrange:
    def __init__(self, ) -> None:
        
        # 初始化位置（x）和速度（v）张量
        self.x = torch.zeros(cf.n_node, 2, dtype=torch.float32)
        self.v = torch.zeros(cf.n_node, 2, dtype=torch.float32)
        self.acc = torch.zeros(cf.n_node, dtype=torch.float32)
        self.triangles = torch.zeros(size=[cf.n_triangles, 3], dtype=torch.int)
        
        self.elements = torch.zeros(size=[cf.n_triangles,2, 2])
        self.elements_Dm_inv = torch.zeros(size=[cf.n_triangles,2, 2])
        self.F = torch.zeros(size=[cf.n_triangles, 2, 2])
        self.element_v0 = torch.zeros(size=[cf.n_triangles])
        self.elements_energy_dens = torch.zeros(size=[cf.n_triangles,])
        self.sig = torch.zeros(size = [cf.n_triangles, 2, 2])

        # 节点质量，重力向量
        self.m =cf.m
        self.g = torch.tensor([0., - cf.g], dtype=torch.float32)
        self.dh = cf.dh

        # 边界约束的索引 左边界固定
        self.boundary_index_left = self.boundary_index_left_func()
        self.boundary_index_right = self.boundary_index_right_func()

        self.init_coordinates()
        self.x_origin = self.x.clone()
        self.init_triangles()
        self.init_elements()

        # 输出文件时将cell data 投影到 point data
        self.num_emergences_averaged = self.cal_num_emergences()
        
    def boundary_index_left_func(self,):
        tmp = []    
        for j in range(cf.ny):
            tmp.append(ij_2_index(0, j))
        return torch.tensor(tmp, dtype=torch.int)
    
    def boundary_index_right_func(self,):
        tmp = []    
        for j in range(cf.ny):
            tmp.append(ij_2_index(cf.nx-1, j))
        return torch.tensor(tmp, dtype=torch.int)

    def init_coordinates(self,):
        for i in range(cf.nx):
            for j in range(cf.ny):
                index = ij_2_index(i, j)
                self.x[index] = torch.tensor([cf.init_x + i*cf.dx, cf.init_y + j*cf.dx], dtype=torch.float32)
                self.v[index] = torch.tensor([0., 0.], dtype=torch.float32)

    def init_triangles(self,):
        for i in range(cf.nx-1):
            for j in range(cf.ny-1):
                # 上三角单元
                tid = (i * (cf.ny - 1) + j) * 2
                self.triangles[tid][0] = ij_2_index(i, j)
                self.triangles[tid][1] = ij_2_index(i + 1, j)
                self.triangles[tid][2] = ij_2_index(i, j + 1)
                # 下三角单元
                tid = (i * (cf.ny - 1) + j) * 2 + 1
                self.triangles[tid][0] = ij_2_index(i, j + 1)
                self.triangles[tid][1] = ij_2_index(i + 1, j + 1)
                self.triangles[tid][2] = ij_2_index(i + 1, j)
    
    def compute_D(self, x):
        elements = torch.stack([
            x[self.triangles[:, 1]] - x[self.triangles[:, 0]],
            x[self.triangles[:, 2]] - x[self.triangles[:, 0]],
            ]).permute(1, 2, 0)
        return elements

    def init_elements(self, ):
        self.elements = self.compute_D(self.x)
        self.elements_Dm_inv = torch.linalg.inv(self.elements)
        self.element_v0 = torch.abs(torch.linalg.det(self.elements))/2.
        self.F = self.cal_F(self.elements)
        self.F.requires_grad = True
        self.sig, self.elements_energy_dens = self.cal_sigma(self.F)

    def cal_F(self, D):
        F = torch.einsum("nij, njk->nik", D, self.elements_Dm_inv)
        return F
    
    def cal_energy(self, F):
        J = torch.linalg.det(F)
        C = torch.einsum("nji, njk->nik", F, F)
        I1 = torch.einsum("nii->n", C)
        energy_single = 0.5 * cf.lameMu* (I1 - 2 - 2*torch.log(J)) + \
            0.5*cf.lameLa * (J-1)**2
        return energy_single * self.element_v0
    
    def cal_sigma(self, F):
        """
            势能方程参考：https://en.wikipedia.org/wiki/Neo-Hookean_solid
        """
        # F.requires_grad= True
        J = torch.linalg.det(F)
        C = torch.einsum("nji, njk->nik", F, F)
        I1 = torch.einsum("nii->n", C)
        energy_single = 0.5 * cf.lameMu* (I1 - 2 - 2*torch.log(J)) + \
            0.5*cf.lameLa * (J-1)**2
        # 计算PK1应力
        P = torch.autograd.grad(
            energy_single, 
            F, 
            grad_outputs= torch.ones_like(energy_single),
            retain_graph=True,
            create_graph=True)[0]
        # 计算柯西应力
        Sig = torch.einsum("n, nij, nkj->nik", 1/J, P, F) 
        return Sig, energy_single * self.element_v0
    
    def train(self,):
        t = 0
        converge_flag = False
        last_internal = 0.
        try_num = 0
        while t < cf.iteration and not converge_flag:
            x_clone = self.x.clone().detach().requires_grad_(True)
            it_time = time.time()
            elements = self.compute_D(x_clone)
            F = self.cal_F(elements)
            energy_density = self.cal_energy(F=F)

            Internal = torch.sum(energy_density)
            # self.optimizer.zero_grad()
            Internal.backward()
            self.acc = -x_clone.grad/self.m + self.g
            self.v += self.acc * 0.5* self.dh if t ==0 else self.acc * self.dh

            # 位移边界条件，将对应节点速度设置为0
            self.v[self.boundary_index_left, :] *= 0.
            # self.v[self.boundary_index_right, :] *= 0.
            # 更新节点
            self.x += self.dh * self.v
            # 位移边界条件
            # factor = min(t, cf.load_step_len)/cf.load_step_len
            # self.x[self.boundary_index_right, :] = self.x_origin[self.boundary_index_right, :] +  \
            #      factor * torch.tensor([cf.known_right_ux, cf.known_right_uy], dtype=torch.float32)

            # damping
            self.v *= np.exp(-self.dh * 50.)

            if (t+1) % cf.save_per_epoch==0 or t==0:
                if last_internal != 0:
                    improvement = abs((Internal.item() - last_internal) / last_internal) 
                else:
                    improvement = 1.0
                if improvement< cf.internal_tolerance \
                    and t>cf.save_per_epoch:
                    try_num += 1
                    if try_num >cf.patience_num:
                        converge_flag = True
                        print("="*60)
                        print(f"Computation converged with internal "
                            f"{abs((Internal.item() - last_internal) / last_internal):.3e} !") 
                else:
                    try_num = 0
                        
                line = 'Iter: %d Internal: %.9e External: %.9e Improvement: %.3e Time: %.3e(s) ' \
                        % (t+1 if t>0 else t, Internal.item(), 0., improvement,
                        (time.time() - it_time))
                self.sig, self.elements_energy_dens = self.cal_sigma(F)
                self.F = F
                print(line)
        
                self.write_to_vtk(epoch=t)

                

                last_internal = Internal.item()
            t+=1

        self.plot()

    def plot(self):
        """

        """
        tmp = self.x.detach().numpy()
        plt.scatter(tmp[:, 0], tmp[:, 1])
        plt.plot([cf.init_x, cf.init_x + cf.dx * (cf.nx-1)], 
                    [cf.init_y, cf.init_y], 'k')
        plt.plot([cf.init_x, cf.init_x + cf.dx * (cf.nx-1)], 
                    [cf.init_y + cf.dx * (cf.ny-1),cf.init_y + cf.dx * (cf.ny-1)], 'k')
        plt.plot([cf.init_x, cf.init_x], 
                    [cf.init_y, cf.init_y + cf.dx * (cf.ny-1)], 'r')
        plt.plot([cf.init_x + cf.dx * (cf.nx-1), cf.init_x + cf.dx * (cf.nx-1)], 
                    [ cf.init_y, cf.init_y + cf.dx * (cf.ny-1),], 'b')
        plt.axis("equal")
        plt.show()
        plt.close()
        
    def write_to_vtk(self, epoch:int):
        x_arr = np.float64(self.x.detach().numpy().reshape(cf.nx, cf.ny, cf.dim))

        sig_node = self.cell_data_to_point(self.sig)
        eps_node = self.cell_data_to_point(0.5 * (torch.einsum("nji, njk->nik", self.F, self.F) - torch.eye(2)))
        
        x_arr_x = np.ascontiguousarray(x_arr[..., 0:1].repeat(2, axis=2).transpose(1, 0, 2))
        x_arr_y = np.ascontiguousarray(x_arr[..., 1:2].repeat(2, axis=2).transpose(1, 0, 2))
        sig_xx = np.ascontiguousarray(sig_node[..., 0, 0, np.newaxis].repeat(2, axis=2).transpose(1, 0, 2))
        sig_xy = np.ascontiguousarray(sig_node[..., 0, 1, np.newaxis].repeat(2, axis=2).transpose(1, 0, 2))
        sig_yy = np.ascontiguousarray(sig_node[..., 1, 1, np.newaxis].repeat(2, axis=2).transpose(1, 0, 2))

        sig_v = np.ascontiguousarray(0.5 * (sig_xx + sig_yy))
        sig_von_mises_tmp = cal_von_miese_stress(
                sig_xx=sig_node[..., 0, 0], sig_xy=sig_node[..., 0, 1], sig_yy = sig_node[..., 1, 1])
        sig_von_mises = np.ascontiguousarray(
            sig_von_mises_tmp[..., np.newaxis].repeat(2, axis=2).transpose(1, 0, 2))

        eps_xx = np.ascontiguousarray(eps_node[..., 0, 0, np.newaxis].repeat(2, axis=2).transpose(1, 0, 2))
        eps_xy = np.ascontiguousarray(eps_node[..., 0, 1, np.newaxis].repeat(2, axis=2).transpose(1, 0, 2))
        eps_yy = np.ascontiguousarray(eps_node[..., 1, 1, np.newaxis].repeat(2, axis=2).transpose(1, 0, 2))
        eps_v  = np.ascontiguousarray(eps_xx + eps_yy)

        gridToVTK(f'{cf.filename_out:s}/output_{epoch+1}', 
                    x_arr_x, x_arr_y, cf.zz, 
                    pointData={"U": (np.ascontiguousarray((x_arr_x - cf.xx)),  
                                     np.ascontiguousarray((x_arr_y - cf.yy)), 
                                     np.zeros_like(x_arr_y)),
                               "Sig_v": sig_v,
                               "Sig_mises": sig_von_mises,
                               "Sig_xx": sig_xx,
                               "Sig_xy": sig_xy,
                               "Sig_yy": sig_yy, 
                               
                               "Eps_v":  eps_v,
                               "Eps_xx": eps_xx,
                               "Eps_xy": eps_xy,
                               "Eps_yy": eps_yy, },
                    )

        # print(f"\t\tVTK file saved as {cf.filename_out:s}/output_{epoch+1}")

    def cell_data_to_point(self, cell_data: torch.Tensor):
        """
            将单元内的数据project到points上
        """
        node_data = np.zeros(shape=[cf.nx * cf.ny, 2,2], dtype=np.float64)
        
        cell_data = np.float64(cell_data.cpu().detach().numpy())

        for i, node_nums in enumerate(self.triangles):
            node_data[node_nums] += cell_data[i]

        node_data = np.einsum(
            "n, nij->nij", self.num_emergences_averaged, node_data).reshape(cf.nx, cf.ny, cf.dim, cf.dim)
        return node_data
    
    def cal_num_emergences(self, ):
        """
            统计每一个点在单元中出现的次数

            用与将cell data project 到 points上的时候求平均
        """
        node_emerge_num = np.zeros(shape=[cf.nx * cf.ny], dtype=np.int8)
      
        for i, node_nums in enumerate(self.triangles):
            node_emerge_num[node_nums] += 1
        num_emergences_averaged = 1/node_emerge_num
        return num_emergences_averaged
    
def main():
    obj = Torch_Lagrange()
    obj.train()

    # 输出结果到vts




if __name__ == "__main__":
    main()