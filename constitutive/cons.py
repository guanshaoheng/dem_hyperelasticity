import torch
import numpy as np
from constitutive.utils_cons import rotation_matrix, read_integration_sphere


class DegradedCons:
    def __init__(
            self, 
            theta: float, phi: float, 
            K: float=500e3, mu: float=62.1e3, c1: float=56.59e3, c2:float=3.83, healthy_ratio: float=1.0, be: float=1.0,
            device=torch.device("cpu") ) -> None:
        """
            用于3D求解, 此处只将弹性纤维进行积分，没有考虑蛋白质纤维
            theta: 纤维的方向 rad 弧度 极角, 0的时候为极轴（z轴）方向，pi/2的时候方向在xy平面内
            phi: 纤维的方向 rad 弧度 方位角
            healthy_ratio: 1.0健康 0.0完全破坏
            K: 体积模量
            mu: 剪切模量
            c1: 弹性纤维参数 
            c2: 弹性纤维参数

            num_pieces: 进行积分时将0.5 pi分成num_pieces份
        """

        # computational device
        self.device = device

        # 材料参数
        self.K = K   # 体积模量 bulk modulus used as the bulk deformation penalty
        self.mu = mu # 剪切模量
        self.c1, self.c2, self.be = c1, c2, be
        self.healthy_ratio = healthy_ratio  # 破坏参数，当healthy_ratio为0则完全破坏，1为健康
        self.healthy_ratio_rad = self.healthy_ratio * np.pi*0.5
        self.theta = torch.tensor(theta, dtype=torch.float32, device=self.device)
        self.phi = torch.tensor(phi, dtype=torch.float32, device=self.device)
        self.kronecker = torch.eye(3, device=self.device)
        # self.p = 0.5 * (torch.einsum("ik, jl->ijkl", self.kronecker, self.kronecker) + 
        #                 torch.einsum("il, jk->ijkl", self.kronecker, self.kronecker)) - \
        #         (torch.einsum("ij, kl->ijkl", self.kronecker, self.kronecker))/3.0

        # 计算纤维方向矩阵
        # self.N = self.get_direction_vector(theta=self.theta, phi=self.phi)
        self.R = rotation_matrix(theta=self.theta, phi=self.phi, device=self.device)
        # self.NxN = torch.ger(self.N, self.N)

        # --- integration configutation --- 
        # read from the integration sphere
        integration_areas, integration_centres = read_integration_sphere()
        self.integration_areas, self.integration_centres = integration_areas.to(self.device), integration_centres.to(self.device) 
        self.healthy_ = torch.where(self.integration_centres[:, 2]> np.cos(self.healthy_ratio_rad), 1., 0.)
        self.weights_of_fiber_ = self.weight_on_spere_cos(cos_theta=self.integration_centres[:, 2])
        self.ds_weights_fiber_ = self.weights_of_fiber_ * self.integration_areas
        self.ds_weights_fiber_normed_ = self.ds_weights_fiber_ / torch.sum(self.ds_weights_fiber_)

    def total_energy_batch(self, F: torch.Tensor):
        """
            F: 一个batch的变形张量 形状为 (batach_size, 3, 3)
        """
        J = torch.det(F)
        # J = torch.einsum("nii->n", F)-2.  # small deformation assumption
        F_ = torch.einsum("n, nij->nij", J ** (-1/3), F)
        C_ = torch.einsum("nji, njk->nik", F_, F_)
        I1_ = torch.einsum("nii->n", C_)

        en_vol = self.get_volumetric_engergy(J=J)
        en_ground = self.get_shear_energy(I1_=I1_)
        en_integrated = self.get_integrated_energy_batch(C_=C_)

        return  en_vol + en_ground + en_integrated
                    
    def get_dev(self, t):
        return t- torch.eye(3)*torch.trace(t)/3.

    def get_volumetric_engergy(self, J):
        return self.K * (J**2 - 1. - 2 * torch.log(J))
    
    def get_shear_energy(self, I1_):
        return 0.5 * self.mu * (I1_ - 3.0)
    

    def get_integrated_energy_batch(self, C_: torch.Tensor)->torch.Tensor:
        r"""
            calculate the energy from the elastic fiber
            C_: 一个batch的C_ 形状为 (batch_size, 3, 3)

            计算在球面的能量积分 根据公式12

            E_R 就是 self.N

            在这里积分的时候, 采用以E_R为[0, 0, 1]轴的球形坐标系中，
            首先将C_从原始坐标系中旋转到该坐标系中, 然后进行积分
            积分域为 \theta \in [0, pi/2], \phi \in [0, 2\pi]

            方位角必须积分360度

            如果取 num_pieces 总共需要计算的点的个数为 4 * num_pices**2
        """
        # 旋转到以纤维方向为z轴的坐标系后的 C_
        # 以纤维的主方向为z轴进行积分
        C_rotated = torch.einsum("ij, njk, lk->nil", self.R, C_, self.R)  # rotate the tensor to the z axis
        I4_ = torch.einsum("qij, ni, nj->qn", C_rotated, self.integration_centres, self.integration_centres)
        energy_density_n_ = torch.where(I4_>=1.0, 1.0, 0.) * self.healthy_ * self.fen(I4_)
        energy_density_ = torch.sum(self.ds_weights_fiber_normed_ * energy_density_n_, axis=[1])
        
        return energy_density_

    def weight_on_spere(self, theta: torch.Tensor)-> torch.Tensor:
            
        return torch.exp(self.be * torch.cos(theta)**2)
    
    def weight_on_spere_cos(self, cos_theta: torch.Tensor)->torch.Tensor:
        return torch.exp(self.be * cos_theta**2)

    def get_engergy_density(self, I4_, theta: float):
        """
            C_ 拉伸的值的平方, 是一个torch tensor, 与网络的位移预测值有关
            theta rad 当前积分点的极角
            self.healthy_ratio rad 破坏的极角, 大于该角的弹性纤维不计算
        """
        if theta<self.healthy_ratio_rad and I4_>1.:
            return self.fen(I4_)
        else: return  0. 
    
    def fen(self, I4_):
        # """
        #     refer to Li's paper Eq. 3.2  &  Eq. 3.3
        #         A discrete fibre dispersion method for excluding fibres 
        #         under compression in the modelling of fibrous tissues

        #         for Eq. 3.3 mu = 5e3 v=10e3 be=2.9

        # """
        # return self.k1/self.k2/2. * torch.exp(self.k2 * (I4_-1) ** 2 -1.)  # Eq. 3.2 
        # return 0.5 * self.mu * (I4_ - 1) ** 2  # Eq. 3.3
        """
            refer to Malte's paper Eq. 35
                A discrete approach for modeling degraded elastic fibers in aortic dissection
            c1 = 56.59e3 c2=3.83 be=0.01

        """
        return self.c1/self.c2 * (I4_**(self.c2*0.5)-1.0) - self.c1 * torch.log(I4_) * 0.5
    
    def get_direction_vector(self, theta: torch.Tensor, phi: torch.Tensor)->torch.Tensor:
        """
            theta: in shape of [num_samples,]
            phi:   in shape of [num_samples,]
        """
        return torch.stack([
            torch.sin(theta) * torch.cos(phi), 
            torch.sin(theta)*torch.sin(phi), 
            torch.cos(theta)]).permute(1, 0)
    
    def get_direction_vector_tensor(self, theta: torch.Tensor, phi: torch.Tensor)->torch.Tensor:
        """
            用来直接计算所有积分点的方向向量
        """
        return torch.stack([
            torch.sin(theta) * torch.cos(phi), 
            torch.sin(theta)*torch.sin(phi), 
            torch.cos(theta)]).permute(1, 2, 0)

    def get_cauchy_stress_batch(self, F: torch.Tensor)->torch.Tensor:
        """
        F: 一个batch的变形梯度 形状为 (batch_size, 3, 3)
            
            计算该点的能量密度和应力张量

            返回值：能量密度, 应力张量
        """
        # F.requires_grad = True  #没有必要加这句话，应为已经要求 grad 了

        # 变形张量和不变量
        J = torch.det(F)
        # J = torch.einsum("nii->n", F)-2.  # small deformation assumption
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

        F_inv = torch.inverse(F)
        pk2 = torch.einsum("nij, njk->nik", F_inv, pk1)
       
        return vol_energy+isochoric_energy, sigma, pk1, pk2
    

class DegradedCons_different_health(DegradedCons):
    """
        include the random field of healthy_ratio;
        healthy ratio: 1 -> totally healthy, 0 -> totally damaged 
    """
    def __init__(
            self, 
            healthy_ratio_tensor: torch.Tensor,
            theta: float, phi: float,
            K: float=500e3, mu: float=62.1e3, c1: float=56.59e3, c2:float=3.83, healthy_ratio: float=1.0,
            device=torch.device("cpu")) -> None:
        super().__init__(theta, phi, K, mu, c1, c2, healthy_ratio, device=device)
        
        self.healthy_ratio_rad = healthy_ratio_tensor.to(self.device) * np.pi * 0.5
        # repeat the input points from shape [num_points, ] to [num_triangle_integrations, num_points]
        tmp = self.integration_centres[:, 2].repeat(len(self.healthy_ratio_rad), 1 ).permute(1, 0)
        # change into shape of [num_points, num_triangle_integrations]
        self.healthy_ = torch.where(tmp > torch.cos(self.healthy_ratio_rad), 1., 0.).permute(1, 0)
        """
        finished: Check if the random field is correctly applied
        from matplotlib import pyplot as plt
        tmp = self.healthy_ratio_rad.detach().numpy().reshape(25, 25, 4) / 0.5/np.pi
        plt.imshow(tmp); plt.colorbar(); plt.show()
        """



class NeoHooking():
    def __init__(self, k: float, mu: float) -> None:
        """
        This simple constitutive model is to check if the stress on the 
        right side boundary equals to the applied traction there.

        Yes, the stress equals to the applied traction on the right bo-
        undary.

        Reference: https://en.wikipedia.org/wiki/Neo-Hookean_solid
        
        Parameteres:
        - k: bulk modulus
        - mu: shear modulus
        """
        self.k = k
        self.mu = mu
        self.lam = self.k - 2.*self.mu / 3.
        self.c1 = 0.5 * self.mu
        self.d1 = 0.5 * self.lam

    def total_energy_batch(self, F: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        - F: the deformation tensor in shape of [num_samples, 3, 3]

        Returns:
        - W: the energy in shape of [num_samples,]
        """
        J = torch.det(F)
        C = torch.einsum("nji, njk->nik", F, F)
        I = torch.einsum("nii->n", C)
        W = self.c1 * (I - 3. - 2.*torch.log(J)) + self.d1 * (J - 1.0)**2
        return W
    
    def get_cauchy_stress_batch(self, F: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        - F

        Returns:
        - W: elastic energy 
        - sigma: Cauchy stress
        - pk1: first Piola-Kirchhoff stress in shape of [num_samples, 3, 3]
        - pk2: second Piola-Kirchhoff stress in shape of [num_samples, 3, 3]
        """
        J = torch.det(F)
        W = self.total_energy_batch(F)
        pk1 = torch.autograd.grad(
            W, F, grad_outputs=torch.ones_like(W), 
            create_graph=True, retain_graph=True)[0]
        
        sigma =  torch.einsum("n, nij, nkj->nik", 1./J, pk1, F)

        F_inv = torch.inverse(F)
        pk2 = torch.einsum("nij, njk->nik", F_inv, pk1)
       
        return W, sigma, pk1, pk2
        
        
    

if __name__ == "__main__":

    # check the basic class (DegradedCons)
    obj = DegradedCons(theta=0.3 * np.pi*0.5, phi=0.23 * np.pi*0.5, healthy_ratio=1.0)
    F = torch.diag(torch.tensor([0.9, 0.9, 1.2], dtype=torch.float32)).reshape(1, 3, 3)
    F.requires_grad = True
    energy = obj.total_energy_batch(F = F)
    energyy, sigma, pk1, pk2 = obj.get_cauchy_stress_batch(F = F.reshape(1, 3, 3))

    #  check the DegradedCons_different_health class
    n_integration_points = 99
    obj_different_health = DegradedCons_different_health(
        healthy_ratio_tensor=torch.rand([n_integration_points]),
        theta=0.5 * np.pi*0.3, phi=0.5 * np.pi*0.23)
    energy_different = obj_different_health.total_energy_batch(F.repeat(n_integration_points, 1, 1))


    print()


        