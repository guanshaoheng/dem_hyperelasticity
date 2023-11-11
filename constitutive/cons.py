import torch
import numpy as np
from constitutive.utils_cons import rotation_matrix


class DegradedCons:
    def __init__(self, theta: float, phi: float, 
                 K: float=8e5, mu: float=62.1e3, c1: float=56.59e3, c2:float=3.83, failure: float=1.0,
                 num_pieces: int=5) -> None:
        """
            用于3D求解, 此处只将弹性纤维进行积分，没有考虑蛋白质纤维
            theta: 纤维的方向 rad 弧度 极角
            phi: 纤维的方向 rad 弧度 方位角
            failure: 1.0健康 0.0完全破坏
            K: 体积模量
            mu: 剪切模量
            c1: 弹性纤维参数 
            c2: 弹性纤维参数

            num_pieces: 进行积分时将0.5 pi分成num_pieces份
        """

        # 材料参数
        self.K = K   # 体积模量 bulk modulus
        self.mu = mu # 剪切模量
        self.c1, self.c2 = c1, c2
        self.failure = failure  # 破坏参数，当failure为0则完全破坏，1为健康
        self.failure_rad = failure * np.pi*0.5
        self.theta = torch.tensor(theta, dtype=torch.float32)
        self.phi = torch.tensor(phi, dtype=torch.float32)
        self.kronecker = torch.eye(3)
        self.p = 0.5 * (torch.einsum("ik, jl->ijkl", self.kronecker, self.kronecker) + 
                        torch.einsum("il, jk->ijkl", self.kronecker, self.kronecker)) - \
                (torch.einsum("ij, kl->ijkl", self.kronecker, self.kronecker))/3.0


        # 计算纤维方向矩阵
        self.N = self.get_direction_vector(theta=self.theta, phi=self.phi)
        self.R = rotation_matrix(theta=self.theta, phi=self.phi)
        # self.NxN = torch.ger(self.N, self.N)

        # 积分计算设置
        self.num_pieces = num_pieces  # 将90度切分为3份，在每份的形心处积分
        num_integration_phi = 4
        self.num_integration_points = self.num_pieces**2*num_integration_phi
        self.dh = 0.5*torch.pi/self.num_pieces
        num_theta = self.num_pieces                      # 90 度
        num_phi = self.num_pieces*num_integration_phi    # 360 度
        self.theta_list = torch.linspace(0.5*self.dh, 0.5 * np.pi-0.5*self.dh, num_theta)
        self.phi_list = torch.linspace(0.5*self.dh, num_integration_phi * 0.5 * np.pi-0.5*self.dh, num_phi)
        self.theta_mesh, self.phi_mesh = torch.meshgrid(self.theta_list, self.phi_list)
        self.healthy = torch.where(self.theta_mesh<self.failure_rad, 1., 0.)
        self.N_integration_points = self.get_direction_vector_tensor(self.theta_mesh, self.phi_mesh)
        self.weights_of_fiber = self.weight_on_spere(theta=self.theta_mesh)
        self.ds_weights_fiber = self.dh**2*torch.sin(self.theta_mesh)*self.weights_of_fiber
        self.norm = torch.sum(self.ds_weights_fiber)

    def total_energy(self, F: torch.Tensor):
        """
            F: 变形张量，形状为 3x3
        """
        J = torch.det(F)
        F_ = J ** (-1/3) * F
        C_ = torch.matmul(F_.T, F_)
        I1_ = torch.trace(C_)

        energy1 = self.get_volumetric_engergy(J=J) 
        energy2 = self.get_shear_energy(I1_=I1_) 
        energy3 =  self.get_integrated_energy(C_=C_)

        return self.get_volumetric_engergy(J=J) + \
                self.get_shear_energy(I1_=I1_) + self.get_integrated_energy(C_=C_)
                    #self.get_integrated_energy(C_=C_)

    def total_energy_batch(self, F: torch.Tensor):
        #TODO
        """
            F: 一个batch的变形张量 形状为 (batach_size, 3, 3)
        """
        J = torch.det(F)
        F_ = torch.einsum("n, nij->nij", J ** (-1/3), F) 
        C_ = torch.einsum("nji, njk->nik", F_, F_) 
        I1_ = torch.einsum("nii->n", C_)

        return self.get_volumetric_engergy(J=J) + self.get_shear_energy(I1_=I1_) + self.get_integrated_energy_batch(C_=C_)
                    

    def get_dev(self, t):
        return t- torch.eye(3)*torch.trace(t)/3.

    def get_volumetric_engergy(self, J):
        return self.K * (J**2 - 1. - 2 * torch.log(J))
    
    def get_shear_energy(self, I1_):
        return self.mu * (I1_ - 3.0) * 0.5
    
    def get_integrated_energy(
            self, 
            C_: torch.Tensor):
        r"""
            计算在球面的能量积分 根据公式12

            E_R 就是 self.N

            在这里积分的时候, 采用以E_R为[0, 0, 1]轴的球形坐标系中，
            首先将C_从原始坐标系中旋转到该坐标系中, 然后进行积分
            积分域为 \theta \in [0, pi/2], \phi \in [0, 2\pi]

            方位角必须积分360度

            如果取 num_pieces 总共需要计算的点的个数为 4 * num_pices**2
        """
        sum = 0.
        # sum_norm = 0.
        # 旋转到以纤维方向为z轴的坐标系后的 C_
        C_rotated = torch.einsum("ij, jk, lk->il", self.R, C_, self.R)
        # check_R = torch.einsum("ij, kj->ik", self.R, self.R)
        for i in self.theta_list[:-1]:
            # NOTE: use the mid points of the square
            theta=i+0.5*self.dh
            weights = self.weight_on_spere(theta)
            ds = weights *self.dh**2*torch.sin(theta)
            for j in self.phi_list[:-1]:
                # NOTE: use the mid points of the square
                phi=j+0.5*self.dh
                N_tmp = self.get_direction_vector(theta=theta, phi=phi)
                # 计算以该方向为纤维纤维方向得到的I4_
                I4_ = torch.einsum("ij, i, j->", C_rotated, N_tmp, N_tmp)
                # 计算能量密度，同时还要乘上概率密度
                sum += self.get_engergy_density(I4_=I4_, theta=theta) * ds
                # sum_norm += ds
        # print(sum_norm)
        # sum /= sum_norm

        """
            norm number 是密度函数在整个积分域算出来的
                  ans         norm   0.3  0.23
            90   5.387       2.3495
            180  429.4894    4.6990
            360  429.4893    9.3979

                  ans         norm   0.3  0.23
            90   10.5193     2.3495
            180  45.7640     4.6990
            360  429.4893    9.3979

        """
        return sum/9.3979
    

    def get_integrated_energy_batch(self, C_: torch.Tensor)->torch.Tensor:
        """
            C_: 一个batch的C_ 形状为 (batch_size, 3, 3)
        """
        C_rotated = torch.einsum("ij, njk, lk->nil", self.R, C_, self.R)
        I4_ = torch.einsum("qij, mni, mnj->qmn", C_rotated, self.N_integration_points, self.N_integration_points)
        energy_density_n = torch.where(I4_>=1.0, 1.0, 0.) * self.healthy * self.fen(I4_)
        energy_density = torch.sum(self.ds_weights_fiber * energy_density_n, axis=[1, 2])
        return energy_density/self.norm

    
    def weight_on_spere(self, theta: torch.Tensor, be:float=1.0)-> torch.Tensor:
            
        return torch.exp(be * torch.cos(theta)**2)

    def get_engergy_density(self, I4_, theta: float):
        """
            C_ 拉伸的值的平方, 是一个torch tensor, 与网络的位移预测值有关
            theta rad 当前积分点的极角
            self.failure rad 破坏的极角, 大于该角的弹性纤维不计算
        """
        if theta<self.failure_rad and I4_>1.:
            return self.fen(I4_)
        else: return  0. 
    
    def fen(self, I4_):
        return self.c1/self.c2 * (I4_**(self.c2*0.5)-1.0) - self.c1 * torch.log(I4_) * 0.5
    
    def get_direction_vector(self, theta: float, phi: float)->torch.Tensor:
        return torch.tensor([
            torch.sin(theta) * torch.cos(phi), 
            torch.sin(theta)*torch.sin(phi), 
            torch.cos(theta)], dtype=torch.float32)
    
    def get_direction_vector_tensor(self, theta: torch.tensor, phi: torch.tensor)->torch.Tensor:
        """
            用来直接计算所有积分点的方向向量
        """
        return torch.stack([
            torch.sin(theta) * torch.cos(phi), 
            torch.sin(theta)*torch.sin(phi), 
            torch.cos(theta)]).permute(1, 2, 0)
    
    def get_cauchy_stress(self, F: torch.Tensor)->torch.Tensor:
        """
            计算该点的能量密度和应力张量

            返回值：能量密度, 应力张量
        """
        F.requires_grad = True  #没有必要加这句话，应为已经要求 grad 了

        # 变形张量和不变量
        J = torch.det(F)
        F_ = J ** (-1/3) * F
        C_ = torch.matmul(F_.T, F_)
        I1_ = torch.trace(C_)

        # 体应变能量和剪应变能量
        vol_energy = self.get_volumetric_engergy(J=J)
        isochoric_energy = self.get_shear_energy(I1_=I1_) + self.get_integrated_energy(C_=C_)
        total_energy  = vol_energy + isochoric_energy

        # # 平均应力
        # """
        #  参考A new constitutive framework for arterial wall mechanics and a comparative study of material models 公式 (8)
        # """
        # p = torch.autograd.grad(
        #     vol_energy, J, grad_outputs=torch.ones_like(vol_energy), 
        #     create_graph=True, retain_graph=True)[0]
        # # 偏应力计算
        # """
        #     参考 A discrete approach for modeling degraded elastic fibers in aortic dissection 公式 (20~26)
        # """
        # denergy_dC_ = 2.*torch.autograd.grad(
        #     isochoric_energy, C_, grad_outputs=torch.ones_like(isochoric_energy), 
        #     create_graph=True, retain_graph=True)[0]
        # sigma_ = torch.einsum("ij, jk, lk->il", F_, denergy_dC_, F_)/J
        # sigma_dev = torch.einsum("ijkl, kl->ij", self.p, sigma_)
        # sigma = p*self.kronecker + sigma_dev

        """
            柯西应力计算
        """
        pk1 = torch.autograd.grad(
            total_energy, F, grad_outputs=torch.ones_like(total_energy), 
            create_graph=True, retain_graph=True)[0]
        sigma_pk1 = 1/J * torch.einsum("ij, kj->ik", pk1, F)


        return total_energy, sigma

    def get_cauchy_stress_batch(self, F: torch.Tensor)->torch.Tensor:
        """
        F: 一个batch的变形梯度 形状为 (batch_size, 3, 3)
            
            计算该点的能量密度和应力张量

            返回值：能量密度, 应力张量
        """
        # F.requires_grad = True  #没有必要加这句话，应为已经要求 grad 了

        # 变形张量和不变量
        J = torch.det(F)
        F_ = torch.einsum("n, nij->nij", J ** (-1/3), F) 
        C_ = torch.einsum("nji, njk->nik", F_, F_) 
        I1_ = torch.einsum("nii->n", C_)

        # 体应变能量和剪应变能量
        vol_energy = self.get_volumetric_engergy(J=J)
        isochoric_energy = self.get_shear_energy(I1_=I1_) + self.get_integrated_energy_batch(C_=C_)
        total_energy  = vol_energy + isochoric_energy
        r"""
            柯西应力计算
            PK第一应力计算  P = \frac{\partial \psi}{\partial F}
            柯西应力       \sigma = \frac{1}{J} P F^T

            参考： https://zhuanlan.zhihu.com/p/270596659
        """
        pk1 = torch.autograd.grad(
            total_energy, F, grad_outputs=torch.ones_like(total_energy), 
            create_graph=True, retain_graph=True)[0]
        sigma =  torch.einsum("n, nij, nkj->nik", 1./J, pk1, F)
       
        return vol_energy+isochoric_energy, sigma
    

if __name__ == "__main__":
    obj = DegradedCons(theta=np.pi*0.3, phi=np.pi*0.23, failure=1.0)
    F = torch.diag(torch.tensor([0.95, 0.95, 1.2], dtype=torch.float32))
    F.requires_grad = True
    energy = obj.total_energy(F = F)
    energyy, sigma = obj.get_cauchy_stress_batch(F = F.reshape(1, 3, 3))
    print()


        