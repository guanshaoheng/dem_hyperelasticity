from import_file import *
from constitutive.cons import DegradedCons, DegradedCons_different_health
from Utility import cal_jacobian


class EnergyModel:
    # ---------------------------------------------------------------------------------------------------------------------------
    def __init__(
            self, 
            energy: str, dim: int, numg: int,
            K_penalty: float, mu: float, c1: float, c2: float, phi: float, theta: float, 
            healthy_ratio: float=None,
            random_health_flag: bool=False, 
            random_health_field: np.ndarray=np.zeros(2),
            device = torch.device("cpu"), 
            ):
        """
        Parameters:
        - energy (str): define the types of the constitutive model
        - dim (int): demensional number
        - numg (int): number of the points which will be put into the forward function
        - K_penalty(float): bulk modulus
        - mu (float): shear modulus
        - c1 (float): coefficients of the integrated fibers
        - c2 (float): coefficients of the integrated fibers
        - phi (float): fiber orientation 
        - theta (float): fiber orientation 
        - healthy_ratio (float): indicator  0->demaged  1-> healthy
        - random_health_index (bool): yes->using the random healthy field
        - random_health_field (np.ndarray): random field values
        """
        self.type = energy
        self.dim = dim
        
        if self.type == 'degraded':
            if not random_health_flag:
                self.cons = DegradedCons(
                    K=K_penalty, mu=mu, 
                    c1=c1, c2=c2,
                    phi=phi, 
                    theta=theta, 
                    healthy_ratio=healthy_ratio, 
                    device=device,
                    )
            else: 
                # finished: check if the random field is correctly applied 
                self.cons = DegradedCons_different_health(
                    healthy_ratio_tensor=torch.tensor(random_health_field).float(),
                    K=K_penalty, mu=mu, c1=c1, c2=c2,
                    phi=phi, theta=theta, 
                    device=device,
                )
        else:
            raise RuntimeError
        self.I = torch.eye(3, device=dev, dtype=torch.float32).view(1, 3, 3).repeat(numg, 1, 1)


    def getStoredEnergy(self, u, x):
        if self.type == 'neohookean':
            if self.dim == 2:
                return self.NeoHookean2D(u, x)
            if self.dim == 3:
                return self.NeoHookean3D(u, x)
        elif self.type == 'mooneyrivlin':
            if self.dim == 2:
                return self.MooneyRivlin2D(u, x)
            if self.dim == 3:
                return self.MooneyRivlin3D(u, x)
        elif self.type == "degraded":
            if self.dim == 3:
                return self.degraded3D(u, x)
        else:
            raise RuntimeError("There is no type of %s" % self.type)

    def MooneyRivlin3D(self, u, x):
        duxdxyz = torch.autograd.grad(u[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duydxyz = torch.autograd.grad(u[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duzdxyz = torch.autograd.grad(u[:, 2].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        Fxx = duxdxyz[:, 0].unsqueeze(1) + 1
        Fxy = duxdxyz[:, 1].unsqueeze(1) + 0
        Fxz = duxdxyz[:, 2].unsqueeze(1) + 0
        Fyx = duydxyz[:, 0].unsqueeze(1) + 0
        Fyy = duydxyz[:, 1].unsqueeze(1) + 1
        Fyz = duydxyz[:, 2].unsqueeze(1) + 0
        Fzx = duzdxyz[:, 0].unsqueeze(1) + 0
        Fzy = duzdxyz[:, 1].unsqueeze(1) + 0
        Fzz = duzdxyz[:, 2].unsqueeze(1) + 1
        detF = Fxx * (Fyy * Fzz - Fyz * Fzy) - Fxy * (Fyx * Fzz - Fyz * Fzx) + Fxz * (Fyx * Fzy - Fyy * Fzx)
        C11 = Fxx ** 2 + Fyx ** 2 + Fzx ** 2
        C12 = Fxx * Fxy + Fyx * Fyy + Fzx * Fzy
        C13 = Fxx * Fxz + Fyx * Fyz + Fzx * Fzz
        C21 = Fxy * Fxx + Fyy * Fyx + Fzy * Fzx
        C22 = Fxy ** 2 + Fyy ** 2 + Fzy ** 2
        C23 = Fxy * Fxz + Fyy * Fyz + Fzy * Fzz
        C31 = Fxz * Fxx + Fyz * Fyx + Fzz * Fzx
        C32 = Fxz * Fxy + Fyz * Fyy + Fzz * Fzy
        C33 = Fxz ** 2 + Fyz ** 2 + Fzz ** 2
        trC = C11 + C22 + C33
        trC2 = C11*C11 + C12*C21 + C13*C31 + C21*C12 + C22*C22 + C23*C32 + C31*C13 + C32*C23 + C33*C33
        I1 = trC
        I2 = 0.5 * (trC*trC - trC2)
        J = detF
        strainEnergy = self.param_c * (J - 1) ** 2 - self.param_d * torch.log(J) + self.param_c1 * (
                I1 - 3) + self.param_c2 * (I2 - 3)
        return strainEnergy


    def MooneyRivlin2D(self, u, x):
        duxdxy = torch.autograd.grad(u[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duydxy = torch.autograd.grad(u[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        Fxx = duxdxy[:, 0].unsqueeze(1) + 1
        Fxy = duxdxy[:, 1].unsqueeze(1) + 0
        Fyx = duydxy[:, 0].unsqueeze(1) + 0
        Fyy = duydxy[:, 1].unsqueeze(1) + 1
        detF = Fxx * Fyy - Fxy * Fyx
        C11 = Fxx * Fxx + Fyx * Fyx
        C12 = Fxx * Fxy + Fyx * Fyy
        C21 = Fxy * Fxx + Fyy * Fyx
        C22 = Fxy * Fxy + Fyy * Fyy
        J = detF
        traceC = C11 + C22
        I1 = traceC
        trace_C2 = C11 * C11 + C12 * C21 + C21 * C12 + C22 * C22
        I2 = 0.5 * (traceC ** 2 - trace_C2)
        strainEnergy = self.param_c * (J - 1) ** 2 - self.param_d * torch.log(J) + self.param_c1 * (I1 - 2) + self.param_c2 * (I2 - 1)
        return strainEnergy

    # ---------------------------------------------------------------------------------------
    # Purpose: calculate Neo-Hookean potential energy in 3D
    # ---------------------------------------------------------------------------------------
    def NeoHookean3D(self, u, x):
        duxdxyz = torch.autograd.grad(u[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duydxyz = torch.autograd.grad(u[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duzdxyz = torch.autograd.grad(u[:, 2].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        Fxx = duxdxyz[:, 0].unsqueeze(1) + 1
        Fxy = duxdxyz[:, 1].unsqueeze(1) + 0
        Fxz = duxdxyz[:, 2].unsqueeze(1) + 0
        Fyx = duydxyz[:, 0].unsqueeze(1) + 0
        Fyy = duydxyz[:, 1].unsqueeze(1) + 1
        Fyz = duydxyz[:, 2].unsqueeze(1) + 0
        Fzx = duzdxyz[:, 0].unsqueeze(1) + 0
        Fzy = duzdxyz[:, 1].unsqueeze(1) + 0
        Fzz = duzdxyz[:, 2].unsqueeze(1) + 1
        detF = Fxx * (Fyy * Fzz - Fyz * Fzy) - Fxy * (Fyx * Fzz - Fyz * Fzx) + Fxz * (Fyx * Fzy - Fyy * Fzx)
        trC = Fxx ** 2 + Fxy ** 2 + Fxz ** 2 + Fyx ** 2 + Fyy ** 2 + Fyz ** 2 + Fzx ** 2 + Fzy ** 2 + Fzz ** 2
        strainEnergy = 0.5 * self.lam * (torch.log(detF) * torch.log(detF)) - self.mu * torch.log(detF) + 0.5 * self.mu * (trC - 3)
        return strainEnergy

    # ---------------------------------------------------------------------------------------
    # Purpose: calculate Neo-Hookean potential energy in 2D
    # ---------------------------------------------------------------------------------------
    def NeoHookean2D(self, u, x):
        duxdxy = torch.autograd.grad(u[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duydxy = torch.autograd.grad(u[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        Fxx = duxdxy[:, 0].unsqueeze(1) + 1
        Fxy = duxdxy[:, 1].unsqueeze(1) + 0
        Fyx = duydxy[:, 0].unsqueeze(1) + 0
        Fyy = duydxy[:, 1].unsqueeze(1) + 1
        detF = Fxx * Fyy - Fxy * Fyx
        trC = Fxx ** 2 + Fxy ** 2 + Fyx ** 2 + Fyy ** 2
        strainEnergy = 0.5 * self.lam * (torch.log(detF) * torch.log(detF)) - self.mu * torch.log(detF) + 0.5 * self.mu * (trC - 2)
        return strainEnergy

    # ---------------------------------------------------------------------------------------
    # Purpose: calculate degraded potential energy in 3D
    # Shaoheng Guan shaohengguan@gmail.com
    # ---------------------------------------------------------------------------------------
    def degraded3D(self, u, x):
        dudx = cal_jacobian(inputs=x, outputs=u)

        div_u = torch.einsum("nii->n", dudx)

        strainEnergy = self.cons.total_energy_batch(dudx + self.I)
        
        return div_u, strainEnergy
    
    def cal_deformation_tensor(self, u, x):
        return cal_jacobian(inputs=x, outputs=u) + self.I