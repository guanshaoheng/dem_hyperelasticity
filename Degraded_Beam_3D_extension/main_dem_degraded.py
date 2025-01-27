"""
@author: sfmt4368 (Simon), texa5140 (Cosmin), minh.nguyen@ikm.uni-hannover.de

         Implements the 3D Hyperelastic beam models (Mooney-Rivlin)
         
         Shaoheng Guan, shaohengguan@gmail.com
         add the degraded model into this framework.

"""

import define_structure as des
from MultiLayerNet import *
import EnergyModel as md
import Utility as util
import config as cf
from IntegrationLoss import *
from EnergyModel import *
import numpy as np
import time
import torch
from torch.autograd import grad 
import os


class DeepEnergyMethod:
    # Instance attributes
    def __init__(self, model, numIntType, energy, dim):
        # self.data = data
        self.model = MultiLayerNet(model[0], model[1], model[2])
        self.model = self.model.to(dev)
        self.intLoss = IntegrationLoss(numIntType, dim)
        self.energy = energy
        # self.post = PostProcessing(energy, dim)
        self.dim = dim

    def train_model(self, shape, dxdydz, data, neumannBC, dirichletBC, iteration, learning_rate):
        x = torch.from_numpy(data).float()
        x = x.to(dev)
        x.requires_grad_(True)
        # get tensor inputs and outputs for boundary conditions
        # -------------------------------------------------------------------------------
        #                             Dirichlet BC 将边界条件相关的值转化成torch张量
        # -------------------------------------------------------------------------------
        dirBC_coordinates = {}  # declare a dictionary
        dirBC_values = {}  # declare a dictionary
        dirBC_penalty = {}
        for i, keyi in enumerate(dirichletBC):
            dirBC_coordinates[i] = torch.from_numpy(dirichletBC[keyi]['coord']).float().to(dev)
            dirBC_values[i] = torch.from_numpy(dirichletBC[keyi]['known_value']).float().to(dev)
            dirBC_penalty[i] = torch.tensor(dirichletBC[keyi]['penalty']).float().to(dev)
        # -------------------------------------------------------------------------------
        #                           Neumann BC 将边界条件相关的值转化成torch张量
        # -------------------------------------------------------------------------------
        neuBC_coordinates = {}  # declare a dictionary
        neuBC_values = {}  # declare a dictionary
        neuBC_penalty = {}
        for i, keyi in enumerate(neumannBC):
            neuBC_coordinates[i] = torch.from_numpy(neumannBC[keyi]['coord']).float().to(dev)
            neuBC_coordinates[i].requires_grad_(True)  # 允许对力边界的坐标值求导
            neuBC_values[i] = torch.from_numpy(neumannBC[keyi]['known_value']).float().to(dev)
            neuBC_penalty[i] = torch.tensor(neumannBC[keyi]['penalty']).float().to(dev)
        # ----------------------------------------------------------------------------------
        # optimizer = torch.optim.Adam(model.parameters())
        # ----------------------------------------------------------------------------------
        # Minimizing loss function (energy and boundary conditions)
        # ----------------------------------------------------------------------------------
        optimizer = torch.optim.LBFGS(self.model.parameters(), lr=learning_rate, max_iter=20)
        start_time = time.time()
        energy_loss_array = []
        boundary_loss_array = []
        loss_array = []
        for t in range(iteration):
            # Zero gradients, perform a backward pass, and update the weights.
            def closure():
                it_time = time.time()
                # ----------------------------------------------------------------------------------
                # Internal Energy
                # ----------------------------------------------------------------------------------
                u_pred = self.getU(x)
                u_pred.double()

                # 不可压缩材料约束, 计算弹性势能密度
                if cf.model_energy == "degraded":
                    div_u, storedEnergy = self.energy.getStoredEnergy(u_pred, x)
                else:
                    storedEnergy = self.energy.getStoredEnergy(u_pred, x)
                # 积分得到内部能量
                internal2 = self.intLoss.lossInternalEnergy(storedEnergy, dx=dxdydz[0], dy=dxdydz[1], dz=dxdydz[2], shape=shape)
                # 外力作工计算 通过力边界条件计算
                external2 = torch.zeros(len(neuBC_coordinates))
                for i, vali in enumerate(neuBC_coordinates): # 遍历力边界条件
                    neu_u_pred = self.getU(neuBC_coordinates[i])  # 力边界上的点的位移
                    fext = torch.bmm((neu_u_pred + neuBC_coordinates[i]).unsqueeze(1), neuBC_values[i].unsqueeze(2))
                    external2[i] = self.intLoss.lossExternalEnergy(fext, dx=dxdydz[1], dy=dxdydz[2], shape=[shape[1], shape[2]])
                bc_u_crit = torch.zeros((len(dirBC_coordinates)))
                for i, vali in enumerate(dirBC_coordinates): # 遍历位移边界条件
                    dir_u_pred = self.getU(dirBC_coordinates[i])
                    bc_u_crit[i] = self.loss_squared_sum(dir_u_pred, dirBC_values[i]) * dirBC_penalty[i]
                # 能量项的损失计算为 内部能量与外力做功的差值
                energy_loss = internal2 - torch.sum(external2)
                # direchlit 边界条件 用来约束施加位移加载条件
                boundary_loss = torch.sum(bc_u_crit)
                external_total = - torch.sum(external2)
                loss = internal2 + external_total + boundary_loss
                if cf.model_energy == "degraded":
                    compression_penalty = self.intLoss.lossInternalEnergy(torch.abs(div_u)*cf.K_penalty, dx=dxdydz[0], dy=dxdydz[1], dz=dxdydz[2], shape=shape) 
                    # loss += compression_penalty
                optimizer.zero_grad()
                loss.backward()
                line = 'Iter: %d Loss: %.9e Internal: %.9e External: %.9e Boundary: %.9e Compression: %.9e Time: %.3e mins' \
                    % (t + 1, loss.item(), internal2.item(), external_total.item(), 
                       boundary_loss.item(), compression_penalty.item(),  (time.time() - it_time)/60.)
                print(line)
                f_outstream.writelines(line + "\n")
                energy_loss_array.append(energy_loss.data)
                boundary_loss_array.append(boundary_loss.data)
                loss_array.append(loss.data)
                return loss
            optimizer.step(closure)
        elapsed = time.time() - start_time
        print('Training time(mins): %.4f' % (elapsed/60))

    def getU(self, x):
        u = self.model(x)
        """ 
            固定边界条件
            U = x[:, 0] * u 确保在 x轴坐标为0的点的位移为0
        """
        Ux = x[:, 0] * u[:, 0]
        Uy = x[:, 0] * u[:, 1]
        Uz = x[:, 0] * u[:, 2]
        Ux = Ux.reshape(Ux.shape[0], 1)
        Uy = Uy.reshape(Uy.shape[0], 1)
        Uz = Uz.reshape(Uz.shape[0], 1)
        u_pred = torch.cat((Ux, Uy, Uz), -1)
        return u_pred

    # --------------------------------------------------------------------------------
    # Evaluate model to obtain:
    # 1. U - Displacement
    # 2. E - Green Lagrange Strain
    # 3. S - 2nd Piola Kirchhoff Stress
    # 4. F - Deformation Gradient
    # Date implement: 20.06.2019
    # --------------------------------------------------------------------------------
    def evaluate_model(self, x, y, z):
        energy_type = self.energy.type
        """
            使用 neohookin 和 MooneyRivlin则使用该模型进行计算
        """
        if energy_type != "degraded":
            mu = self.energy.mu
            lmbda = self.energy.lam
            dim = self.dim
            Nx = len(x)
            Ny = len(y)
            Nz = len(z)
            xGrid, yGrid, zGrid = np.meshgrid(x, y, z)
            x1D = xGrid.flatten()
            y1D = yGrid.flatten()
            z1D = zGrid.flatten()
            xyz = np.concatenate((np.array([x1D]).T, np.array([y1D]).T, np.array([z1D]).T), axis=-1)
            xyz_tensor = torch.from_numpy(xyz).float()
            xyz_tensor = xyz_tensor.to(dev)
            xyz_tensor.requires_grad_(True)
            # u_pred_torch = self.model(xyz_tensor)
            u_pred_torch = self.getU(xyz_tensor)
            duxdxyz = grad(u_pred_torch[:, 0].unsqueeze(1), xyz_tensor, torch.ones(xyz_tensor.size()[0], 1, device=dev),
                            create_graph=True, retain_graph=True)[0]
            duydxyz = grad(u_pred_torch[:, 1].unsqueeze(1), xyz_tensor, torch.ones(xyz_tensor.size()[0], 1, device=dev),
                            create_graph=True, retain_graph=True)[0]
            duzdxyz = grad(u_pred_torch[:, 2].unsqueeze(1), xyz_tensor, torch.ones(xyz_tensor.size()[0], 1, device=dev),
                            create_graph=True, retain_graph=True)[0]
            F11 = duxdxyz[:, 0].unsqueeze(1) + 1
            F12 = duxdxyz[:, 1].unsqueeze(1) + 0
            F13 = duxdxyz[:, 2].unsqueeze(1) + 0
            F21 = duydxyz[:, 0].unsqueeze(1) + 0
            F22 = duydxyz[:, 1].unsqueeze(1) + 1
            F23 = duydxyz[:, 2].unsqueeze(1) + 0
            F31 = duzdxyz[:, 0].unsqueeze(1) + 0
            F32 = duzdxyz[:, 1].unsqueeze(1) + 0
            F33 = duzdxyz[:, 2].unsqueeze(1) + 1
            detF = F11 * (F22 * F33 - F23 * F32) - F12 * (F21 * F33 - F23 * F31) + F13 * (F21 * F32 - F22 * F31)
            invF11 = (F22 * F33 - F23 * F32) / detF
            invF12 = -(F12 * F33 - F13 * F32) / detF
            invF13 = (F12 * F23 - F13 * F22) / detF
            invF21 = -(F21 * F33 - F23 * F31) / detF
            invF22 = (F11 * F33 - F13 * F31) / detF
            invF23 = -(F11 * F23 - F13 * F21) / detF
            invF31 = (F21 * F32 - F22 * F31) / detF
            invF32 = -(F11 * F32 - F12 * F31) / detF
            invF33 = (F11 * F22 - F12 * F21) / detF
            C11 = F11 ** 2 + F21 ** 2 + F31 ** 2
            C12 = F11 * F12 + F21 * F22 + F31 * F32
            C13 = F11 * F13 + F21 * F23 + F31 * F33
            C21 = F12 * F11 + F22 * F21 + F32 * F31
            C22 = F12 ** 2 + F22 ** 2 + F32 ** 2
            C23 = F12 * F13 + F22 * F23 + F32 * F33
            C31 = F13 * F11 + F23 * F21 + F33 * F31
            C32 = F13 * F12 + F23 * F22 + F33 * F32
            C33 = F13 ** 2 + F23 ** 2 + F33 ** 2
            E11 = 0.5 * (C11 - 1)
            E12 = 0.5 * C12
            E13 = 0.5 * C13
            E21 = 0.5 * C21
            E22 = 0.5 * (C22 - 1)
            E23 = 0.5 * C23
            E31 = 0.5 * C31
            E32 = 0.5 * C32
            E33 = 0.5 * (C33 - 1)
            if energy_type == 'neohookean' and dim == 3:
                P11 = mu * F11 + (lmbda * torch.log(detF) - mu) * invF11
                P12 = mu * F12 + (lmbda * torch.log(detF) - mu) * invF21
                P13 = mu * F13 + (lmbda * torch.log(detF) - mu) * invF31
                P21 = mu * F21 + (lmbda * torch.log(detF) - mu) * invF12
                P22 = mu * F22 + (lmbda * torch.log(detF) - mu) * invF22
                P23 = mu * F23 + (lmbda * torch.log(detF) - mu) * invF32
                P31 = mu * F31 + (lmbda * torch.log(detF) - mu) * invF13
                P32 = mu * F32 + (lmbda * torch.log(detF) - mu) * invF23
                P33 = mu * F33 + (lmbda * torch.log(detF) - mu) * invF33
            else:
                print("This energy model will be implemented later !!!")
                exit()
            S11 = invF11 * P11 + invF12 * P21 + invF13 * P31
            S12 = invF11 * P12 + invF12 * P22 + invF13 * P32
            S13 = invF11 * P13 + invF12 * P23 + invF13 * P33
            S21 = invF21 * P11 + invF22 * P21 + invF23 * P31
            S22 = invF21 * P12 + invF22 * P22 + invF23 * P32
            S23 = invF21 * P13 + invF22 * P23 + invF23 * P33
            S31 = invF31 * P11 + invF32 * P21 + invF33 * P31
            S32 = invF31 * P12 + invF32 * P22 + invF33 * P32
            S33 = invF31 * P13 + invF32 * P23 + invF33 * P33
            u_pred = u_pred_torch.detach().cpu().numpy()
            F11_pred = F11.detach().cpu().numpy()
            F12_pred = F12.detach().cpu().numpy()
            F13_pred = F13.detach().cpu().numpy()
            F21_pred = F21.detach().cpu().numpy()
            F22_pred = F22.detach().cpu().numpy()
            F23_pred = F23.detach().cpu().numpy()
            F31_pred = F31.detach().cpu().numpy()
            F32_pred = F32.detach().cpu().numpy()
            F33_pred = F33.detach().cpu().numpy()
            E11_pred = E11.detach().cpu().numpy()
            E12_pred = E12.detach().cpu().numpy()
            E13_pred = E13.detach().cpu().numpy()
            E21_pred = E21.detach().cpu().numpy()
            E22_pred = E22.detach().cpu().numpy()
            E23_pred = E23.detach().cpu().numpy()
            E31_pred = E31.detach().cpu().numpy()
            E32_pred = E32.detach().cpu().numpy()
            E33_pred = E33.detach().cpu().numpy()
            S11_pred = S11.detach().cpu().numpy()
            S12_pred = S12.detach().cpu().numpy()
            S13_pred = S13.detach().cpu().numpy()
            S21_pred = S21.detach().cpu().numpy()
            S22_pred = S22.detach().cpu().numpy()
            S23_pred = S23.detach().cpu().numpy()
            S31_pred = S31.detach().cpu().numpy()
            S32_pred = S32.detach().cpu().numpy()
            S33_pred = S33.detach().cpu().numpy()
            surUx = u_pred[:, 0].reshape(Ny, Nx, Nz)
            surUy = u_pred[:, 1].reshape(Ny, Nx, Nz)
            surUz = u_pred[:, 2].reshape(Ny, Nx, Nz)
            surE11 = E11_pred.reshape(Ny, Nx, Nz)
            surE12 = E12_pred.reshape(Ny, Nx, Nz)
            surE13 = E13_pred.reshape(Ny, Nx, Nz)
            surE21 = E21_pred.reshape(Ny, Nx, Nz)
            surE22 = E22_pred.reshape(Ny, Nx, Nz)
            surE23 = E23_pred.reshape(Ny, Nx, Nz)
            surE31 = E31_pred.reshape(Ny, Nx, Nz)
            surE32 = E32_pred.reshape(Ny, Nx, Nz)
            surE33 = E33_pred.reshape(Ny, Nx, Nz)
            surS11 = S11_pred.reshape(Ny, Nx, Nz)
            surS12 = S12_pred.reshape(Ny, Nx, Nz)
            surS13 = S13_pred.reshape(Ny, Nx, Nz)
            surS21 = S21_pred.reshape(Ny, Nx, Nz)
            surS22 = S22_pred.reshape(Ny, Nx, Nz)
            surS23 = S23_pred.reshape(Ny, Nx, Nz)
            surS31 = S31_pred.reshape(Ny, Nx, Nz)
            surS32 = S32_pred.reshape(Ny, Nx, Nz)
            surS33 = S33_pred.reshape(Ny, Nx, Nz)
            SVonMises = np.float64(
                np.sqrt(0.5 * ((surS11 - surS22) ** 2 + (surS22 - surS33) ** 2 + (surS33 - surS11) ** 2 + 6 * (
                        surS12 ** 2 + surS23 ** 2 + surS31 ** 2))))
            U = (np.float64(surUx), np.float64(surUy), np.float64(surUz))
            return U, np.float64(surS11), np.float64(surS12), np.float64(surS13), np.float64(surS22), np.float64(surS23), \
                    np.float64(surS33), np.float64(surE11), np.float64(surE12), \
                    np.float64(surE13), np.float64(surE22), np.float64(surE23), np.float64(surE33), np.float64(SVonMises), \
                    np.float64(F11_pred), np.float64(F12_pred), np.float64(F13_pred), \
                    np.float64(F21_pred), np.float64(F22_pred), np.float64(F23_pred), \
                    np.float64(F31_pred), np.float64(F32_pred), np.float64(F33_pred)
    
    # --------------------------------------------------------------------------------
    # Evaluate model to obtain:
    # 1. U - Displacement
    # 2. E - Green Lagrange Strain
    # 3. S - 2nd Piola Kirchhoff Stress
    # 4. F - Deformation Gradient
    def evaluate_model_degraded(self, x, y, z):
        Nx = len(x)
        Ny = len(y)
        Nz = len(z)
        xGrid, yGrid, zGrid = np.meshgrid(x, y, z)
        x1D = xGrid.flatten()
        y1D = yGrid.flatten()
        z1D = zGrid.flatten()
        xyz = np.concatenate((np.array([x1D]).T, np.array([y1D]).T, np.array([z1D]).T), axis=-1)
        xyz_tensor = torch.from_numpy(xyz).float()
        xyz_tensor = xyz_tensor.to(dev)
        xyz_tensor.requires_grad_(True)
        # u_pred_torch = self.model(xyz_tensor)
        u_pred_torch = self.getU(xyz_tensor)

        F = self.energy.cal_deformation_tensor(u=u_pred_torch, x=xyz_tensor)
        _, S = self.energy.cons.get_cauchy_stress_batch(F)
        E = torch.einsum("nji, njk->nik", F, F) - torch.eye(3, device=dev, dtype=torch.float32).view(1, 3, 3).repeat(Nx*Ny*Nz, 1, 1)

        F11, F12 , F13, F21, F22 , F23, F31, F32 , F33 = self.transfer_tensor_into_vector(F, Nx, Ny, Nz)
        S11, S12 , S13, S21, S22 , S23, S31, S32 , S33 = self.transfer_tensor_into_vector(S, Nx, Ny, Nz)
        E11, E12 , E13, E21, E22 , E23, E31, E32 , E33 = self.transfer_tensor_into_vector(E, Nx, Ny, Nz)

        SVonMises = np.sqrt(0.5 * ((S11 - S22) ** 2 + (S22 - S33) ** 2 + (S33 - S11) ** 2 + 6 * (
                        S12 ** 2 + S23 ** 2 + S31 ** 2)))

        U = (
            np.float64(u_pred_torch[:, 0].detach().cpu().numpy().reshape(Nx, Ny, Nz)),
            np.float64(u_pred_torch[:, 1].detach().cpu().numpy().reshape(Nx, Ny, Nz)),
            np.float64(u_pred_torch[:, 2].detach().cpu().numpy().reshape(Nx, Ny, Nz))
        )

        return  U, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises, F11, F12, F13, F21, F22, F23, F31, F32, F33
    
    def transfer_tensor_into_vector(self, tns, Nx, Ny, Nz):
        """
            张量的形状为 (num_samples, 3, 3)
            vec 的形状为 (9, Nx, Ny, Nz)
        """
        F11 = np.float64(tns[:, 0, 0].detach().cpu().numpy().reshape(Nx, Ny, Nz))
        F12 = np.float64(tns[:, 0, 1].detach().cpu().numpy().reshape(Nx, Ny, Nz))
        F13 = np.float64(tns[:, 0, 2].detach().cpu().numpy().reshape(Nx, Ny, Nz))
        F21 = np.float64(tns[:, 1, 0].detach().cpu().numpy().reshape(Nx, Ny, Nz))
        F22 = np.float64(tns[:, 1, 1].detach().cpu().numpy().reshape(Nx, Ny, Nz))
        F23 = np.float64(tns[:, 1, 2].detach().cpu().numpy().reshape(Nx, Ny, Nz))
        F31 = np.float64(tns[:, 2, 0].detach().cpu().numpy().reshape(Nx, Ny, Nz))
        F32 = np.float64(tns[:, 2, 1].detach().cpu().numpy().reshape(Nx, Ny, Nz))
        F33 = np.float64(tns[:, 2, 2].detach().cpu().numpy().reshape(Nx, Ny, Nz))
        return [F11, F12 , F13, F21, F22 , F23, F31, F32 , F33 ]

    # --------------------------------------------------------------------------------
    # method: loss sum for the energy part
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


if __name__ == '__main__':
    # ----------------------------------------------------------------------
    #                   STEP 1: SETUP DOMAIN - COLLECT CLEAN DATABASE
    # ----------------------------------------------------------------------
    dom, boundary_neumann, boundary_dirichlet = des.setup_domain()
    x, y, z, datatest = des.get_datatest()
    # ----------------------------------------------------------------------
    #                   STEP 2: SETUP MODEL
    # ----------------------------------------------------------------------
    mat = md.EnergyModel(cf.model_energy, 3)
    dem = DeepEnergyMethod([cf.D_in, cf.H, cf.D_out], cf.integration_method, mat, 3)
    # ----------------------------------------------------------------------
    #                   STEP 3: TRAINING MODEL
    # ----------------------------------------------------------------------
    start_time = time.time()
    filename_out = f"./output/dem/{cf.filename_out:s}"
    if not os.path.exists(filename_out):
        os.mkdir(filename_out)
    f_outstream = open(os.path.join(filename_out, "training_history.txt"), 'w')

    dem.train_model(cf.shape, cf.dxdydz, dom, boundary_neumann, boundary_dirichlet, cf.iteration, cf.lr)
    end_time = (time.time() - start_time)/60.
    

    """
        使用Neohookin和MooneyRivlin模型, 与使用Degraded模型的应力计算方法不同
    """
    if mat.type != "degraded":
        U, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises, F11, F12, F13, F21, F22, F23, F31, F32, F33 = dem.evaluate_model(x, y, z)
        
    else: # 使用degraded模型计算
        U, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises, F11, F12, F13, F21, F22, F23, F31, F32, F33 = dem.evaluate_model_degraded(x, y, z)

    util.write_vtk_v2(os.path.join(filename_out, cf.filename_out),
                       x, y, z, U, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises)
    surUx, surUy, surUz = U
    L2norm = util.getL2norm(surUx, surUy, surUz, len(x), len(y), len(z), x[1] - x[0], y[1] - y[0], z[1] - z[0])
    H10norm = util.getH10norm(F11, F12, F13, F21, F22, F23, F31, F32, F33, len(x), len(y), len(z), x[1] - x[0], y[1] - y[0], z[1] - z[0])

    

    # --------------------------------------------------ECHO---------------------------------------------
    print("#"*80)
    line = "\n" + "#"*80 + "\n" + "End time(mins): %.2f" % end_time + "\n" + \
           "L2 norm = %.10f" % L2norm + "\n" + "H10 norm = %.10f" % H10norm + "\n" + "#"*80 + "\n"
    print(line)
    print(filename_out)
    print("#"*80)
    
    f_outstream.writelines(line)
    f_outstream.close()



