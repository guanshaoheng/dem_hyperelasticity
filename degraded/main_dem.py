"""
@author:          
         Shaoheng Guan, shaohengguan@gmail.com
         add the degraded model into this framework.

"""

import define_structure as des
from MultiLayerNet_rotation import MultiLayerNet_curl
import EnergyModel as md
import Utility as util
from IntegrationLoss import *
import numpy as np
import time
import torch
from torch.autograd import grad 
import os
from utils import get_random_field, write_to_vtk


class DeepEnergyMethod:
    # Instance attributes
    def __init__(self, model, numIntType, energy, dim, device=torch.device("cpu")):
        self.device = device
        # self.data = data
        self.model = MultiLayerNet_curl(model[0], model[1], model[2], num_layers=num_layers)
        self.model = self.model.to(self.device)
        self.intLoss = IntegrationLoss(numIntType, dim)
        self.energy = energy
        # self.post = PostProcessing(energy, dim)
        self.dim = dim

        self.external_node_force_right = self.get_node_force_right()

    def train_model(self, shape, dxdydz, data, neumannBC, dirichletBC, iteration, learning_rate):
        x = torch.from_numpy(data).float()
        x = x.to(self.device)
        x.requires_grad_(True)
        # get tensor inputs and outputs for boundary conditions
        # -------------------------------------------------------------------------------
        #                         Dirichlet BC (transfer the values to tensors)
        # -------------------------------------------------------------------------------
        dirBC_coordinates = {}  # declare a dictionary
        dirBC_values = {}  # declare a dictionary
        dirBC_penalty = {}
        for i, keyi in enumerate(dirichletBC):
            dirBC_coordinates[i] = torch.from_numpy(dirichletBC[keyi]['coord']).float().to(self.device).requires_grad_(True)
            dirBC_values[i] = torch.from_numpy(dirichletBC[keyi]['known_value']).float().to(self.device)
            dirBC_penalty[i] = torch.tensor(dirichletBC[keyi]['penalty']).float().to(self.device)
        # -------------------------------------------------------------------------------
        #                         Neumann BC (transfer the values to tensors)
        # -------------------------------------------------------------------------------
        neuBC_coordinates = {}  # declare a dictionary
        neuBC_values = {}  # declare a dictionary
        neuBC_penalty = {}
        for i, keyi in enumerate(neumannBC):
            neuBC_coordinates[i] = torch.from_numpy(neumannBC[keyi]['coord']).float().to(self.device)
            neuBC_coordinates[i].requires_grad_(True)  # 允许对力边界的坐标值求导
            neuBC_values[i] = torch.from_numpy(neumannBC[keyi]['known_value']).float().to(self.device)
            neuBC_penalty[i] = torch.tensor(neumannBC[keyi]['penalty']).float().to(self.device)
        # ----------------------------------------------------------------------------------
        # optimizer = torch.optim.Adam(model.parameters())
        # ----------------------------------------------------------------------------------
        # Minimizing loss function (energy and boundary conditions)
        # ----------------------------------------------------------------------------------
        # optimizer = torch.optim.LBFGS(self.model.parameters(), lr=learning_rate, max_iter=20)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        start_time = time.time()
        energy_loss_array = []
        boundary_loss_array = []
        loss_array = []
        loss_min, trial_num = 1e9, 0
        converged_flag = False
        for t in range(iteration):
            # Zero gradients, perform a backward pass, and update the weights.
            it_time = time.time()
            # ----------------------------------------------------------------------------------
            # Internal Energy
            # ----------------------------------------------------------------------------------
            u_pred = self.getU(x)
            u_pred.double()

            # 不可压缩材料约束, 计算弹性势能密度
            if model_energy_type == "degraded":
                div_u, storedEnergy = self.energy.getStoredEnergy(u_pred, x)
            else:
                storedEnergy = self.energy.getStoredEnergy(u_pred, x)
            # 积分得到内部能量
            internal2 = self.intLoss.lossInternalEnergy(storedEnergy, dx=dxdydz[0], dy=dxdydz[1], dz=dxdydz[2], shape=shape)
            # 外力作工计算 通过力边界条件计算
            external2 = torch.zeros(len(neuBC_coordinates))
            for i, vali in enumerate(neuBC_coordinates): # 遍历力边界条件
                neu_u_pred = self.getU(neuBC_coordinates[i])  # 力边界上的点的位移

                # pressures = neuBC_values[i]  # this the pressure which has to be changed into the force
                # force_on_every_square = pressures[0] * hy * hz
                # neu_u_pred_reshape = neu_u_pred.reshape(Ny,  Nz, 3)
                # averaged_summed_u_of_square = 0.25 * (
                #     1.0 * (neu_u_pred_reshape[0, 0] + neu_u_pred_reshape[0, -1] + neu_u_pred_reshape[-1, 0] + neu_u_pred_reshape[-1, -1] ) +
                #     2.0 * (torch.sum(neu_u_pred_reshape[0, 1:-1] + neu_u_pred_reshape[-1, 1:-1], dim=0) + 
                #            torch.sum(neu_u_pred_reshape[1:-1, 0] + neu_u_pred_reshape[1:-1, -1], dim=0))+
                #     4.0 * torch.sum(neu_u_pred_reshape[1:-1, 1:-1], dim=[0, 1])
                # )
                # tmp= torch.sum(averaged_summed_u_of_square * force_on_every_square)

                external2[i] = torch.sum(neu_u_pred * self.external_node_force_right)
            
            # 位移边界条件  Essential boundary (Direchilit boundary)
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
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t % train_interval == 0:
                if t % (10*train_interval) ==0:
                    _, _  = output_results(epoch=t)
                if loss.item()<loss_min:
                    improved_flag = True
                    improve = (loss_min-loss.item())/abs(loss_min)
                    loss_min = loss.item()
                else:
                    improved_flag = False
                    improve = -1

                if improve<improvement_tolerance:
                    trial_num += 1
                else:
                    trial_num = 0
                
                if trial_num>patience:
                    converged_flag = True

                line = 'Iter: %d Loss: %.9e Internal: %.9e External: %.9e Boundary: %.9e Time: %.3e(mins) improve: %.1e trial: %d/%d' \
                    % (t , loss.item(), internal2.item(), external_total.item(), 
                    boundary_loss.item(), (time.time() - it_time)/60., improve, trial_num, patience)
                print(line)
                f_outstream.writelines(line + "\n")
                energy_loss_array.append(energy_loss.data)
                boundary_loss_array.append(boundary_loss.data)
                loss_array.append(loss.data)
                
                if converged_flag:
                    break
        
        # --------------------------- ECHO ---------------------------
        L2norm, H10norm = output_results(epoch = t)  # save the final results
        end_time = (time.time() - start_time)/60.
        print("#"*80)
        line = "\n" + "#"*80 + "\n" + "End time(mins): %.2f" % end_time + "\n" + \
            "L2 norm = %.10f" % L2norm + "\n" + "H10 norm = %.10f" % H10norm + "\n" + "#"*80 + "\n"
        print(line)
        f_outstream.writelines(line)
        f_outstream.close()
        elapsed = time.time() - start_time
        print('Training time(mins): %.4f' % (elapsed/60))
        return 
    
    def get_node_force_right(self,):
        # dic_neumann = boundary_neumann["neumann_1"]
        # coord = dic_neumann["coord"]
        # know_value = dic_neumann["known_value"]
        # penalty = dic_neumann["penalty"]
        # tmp = coord.reshape(Ny, Nz, 3)
        force_on_square = torch.tensor(
            [known_right_tx, known_right_ty, known_right_tz], 
            dtype=torch.float, device=self.device)* hy * hz
        node_force = force_on_square.repeat(Ny, Nz, 1)
        node_force[0, 0] *= 0.25
        node_force[0, -1] *= 0.25
        node_force[-1, 0] *= 0.25
        node_force[-1, -1] *= 0.25
        node_force[1:-1, 0] *= 0.5   # up
        node_force[1:-1, -1] *= 0.5  # down
        node_force[0, 1:-1] *= 0.5   # left
        node_force[-1, 1:-1] *= 0.5  # right
        
        return node_force.reshape(-1, 3)

    def getU(self, x):
        """ 
            fix the left side of the simulation domain
            U = x[:, 0] * u  constrain the displacement of left side to be 0
        """
        # u = self.model.forward_curl(x)
        u = torch.einsum("nj, n->nj", self.model.forward(x), x[:, 0])
        # u = torch.einsum("nj, n->nj", self.model.forward_curl(x), x[:, 0]**2)
        return u
    
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
        xyz = np.concatenate(
            (xGrid.flatten()[:, np.newaxis], 
            yGrid.flatten()[:, np.newaxis], 
            zGrid.flatten()[:, np.newaxis]), axis=1)
        # xyz = np.concatenate((np.array([x1D]).T, np.array([y1D]).T, np.array([z1D]).T), axis=-1)
        xyz_tensor = torch.from_numpy(xyz).float()
        xyz_tensor = xyz_tensor.to(self.device)
        xyz_tensor.requires_grad_(True)
        # u_pred_torch = self.model(xyz_tensor)
        u_pred_torch = self.getU(xyz_tensor)

        F = self.energy.cal_deformation_tensor(u=u_pred_torch, x=xyz_tensor)
        _, S, pk1, pk2 = self.energy.cons.get_cauchy_stress_batch(F)  # vol_energy+isochoric_energy, sigma, pk1, pk2
        E = 0.5 *(torch.einsum("nji, njk->nik", F, F) - torch.eye(3, device=device, dtype=torch.float32).view(1, 3, 3).repeat(Nx*Ny*Nz, 1, 1))
        E_cauchy = 0.5 * (F +  torch.einsum("nij->nji", F))  - torch.eye(3, device=device, dtype=torch.float32).view(1, 3, 3).repeat(Nx*Ny*Nz, 1, 1)

        F11, F12 , F13, F21, F22 , F23, F31, F32 , F33 = self.transfer_tensor_into_vector(F, Nx, Ny, Nz)
        S11, S12 , S13, S21, S22 , S23, S31, S32 , S33 = self.transfer_tensor_into_vector(S, Nx, Ny, Nz)
        E11, E12 , E13, E21, E22 , E23, E31, E32 , E33 = self.transfer_tensor_into_vector(E, Nx, Ny, Nz)
        pk1_11, pk1_12 , pk1_13, pk1_21, pk1_22 , pk1_23, pk1_31, pk1_32 , pk1_33 = self.transfer_tensor_into_vector(pk1, Nx, Ny, Nz)
        pk2_11, pk2_12 , pk2_13, pk2_21, pk2_22 , pk2_23, pk2_31, pk2_32 , pk2_33 = self.transfer_tensor_into_vector(pk2, Nx, Ny, Nz)

        E11_cauchy, E12_cauchy , E13_cauchy, E21_cauchy,\
            E22_cauchy , E23_cauchy, E31_cauchy, E32_cauchy ,\
            E33_cauchy = self.transfer_tensor_into_vector(E_cauchy, Nx, Ny, Nz)

        SVonMises = np.sqrt(0.5 * ((S11 - S22) ** 2 + (S22 - S33) ** 2 + (S33 - S11) ** 2 + 6 * (
                        S12 ** 2 + S23 ** 2 + S31 ** 2)))

        U = (
            np.float64(u_pred_torch[:, 0].detach().cpu().numpy().reshape(Nx, Ny, Nz)),
            np.float64(u_pred_torch[:, 1].detach().cpu().numpy().reshape(Nx, Ny, Nz)),
            np.float64(u_pred_torch[:, 2].detach().cpu().numpy().reshape(Nx, Ny, Nz))
        )

        return  U, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises, F11, F12, F13, F21, F22, F23, F31, F32, F33, \
            E11_cauchy, E12_cauchy, E13_cauchy, E22_cauchy, E23_cauchy, E33_cauchy, \
            pk1_11, pk1_12 , pk1_13, pk1_21, pk1_22 , pk1_23, pk1_31, pk1_32 , pk1_33, \
            pk2_11, pk2_12 , pk2_13, pk2_21, pk2_22 , pk2_23, pk2_31, pk2_32 , pk2_33
    
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
    

def output_results(epoch):
    U, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, \
        SVonMises, F11, F12, F13, F21, F22, F23, F31, F32, F33, \
        E11_cauchy, E12_cauchy, E13_cauchy, E22_cauchy, E23_cauchy, E33_cauchy, \
            pk1_11, pk1_12 , pk1_13, pk1_21, pk1_22 , pk1_23, pk1_31, pk1_32 , pk1_33, \
            pk2_11, pk2_12 , pk2_13, pk2_21, pk2_22 , pk2_23, pk2_31, pk2_32 , pk2_33 = dem.evaluate_model_degraded(x, y, z)

    write_to_vtk(
        filename_out, epoch,
        x, y, z, U, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises, 
        E11_cauchy, E12_cauchy, E13_cauchy, E22_cauchy, E23_cauchy, E33_cauchy,
        pk1_11, pk1_12 , pk1_13, pk1_21, pk1_22 , pk1_23, pk1_31, pk1_32 , pk1_33, 
        pk2_11, pk2_12 , pk2_13, pk2_21, pk2_22 , pk2_23, pk2_31, pk2_32 , pk2_33,
        random_field=None,
        )
    surUx, surUy, surUz = U
    L2norm = util.getL2norm(surUx, surUy, surUz, len(x), len(y), len(z), x[1] - x[0], y[1] - y[0], z[1] - z[0])
    H10norm = util.getH10norm(F11, F12, F13, F21, F22, F23, F31, F32, F33, len(x), len(y), len(z), x[1] - x[0], y[1] - y[0], z[1] - z[0])
    return L2norm, H10norm


if __name__ == '__main__':
    # calculation settings
    device = util.get_torch_device()
    improvement_tolerance = 1e-3
    patience = 10
    train_interval = 100
    # ----------------------------------------------------------------------
    # network settings 
    iteration = int(1e5) + 1 # 2e3
    lr = 0.5
    D_in = 3
    H = 30
    D_out = 3
    num_layers = 4
    # material parameter 
    K_penalty = 500e3 # 60e3
    mu = 62.1e3 * 0.1
    c1 = 56.59e3
    c2 = 3.83
    model_energy_type = 'degraded' # neohookean  degraded
    
    theta_ratio = 1.0   # the fiber direction # 0., 0.2, 0.4, 0.6, 0.8, 1.0
    phi_ratio = 0.      # the fiber direction in x-z plane
    health_coefficient = 1.  # 1.0 表示完全健康，0.0表示完全损坏
    integration_method = "simpson"  # simpson trapezoidal 积分计算方法 
    # define structural parameters 
    Length = lx = 2
    Height = ly = 2
    Depth = lz= 0.25
    known_left_ux = 0
    known_left_uy = 0
    known_left_uz = 0
    bc_left_penalty = 1000000.

    known_right_tx = 7.5e4 * 0.5
    known_right_ty = 0.
    known_right_tz = 0.
    bc_right_penalty = 1.0
    # ------------------------------ define domain and collocation points -------------------------------
    Nx = int(24) + 1  # 120  # 120
    Ny = int(24) + 1  # 30   # 60
    Nz = int(3)  + 1  # 30   # 10
    numg = Nx*Ny*Nz
    x_min, y_min, z_min = (0.0, 0.0, 0.0)
    (hx, hy, hz) = (Length / (Nx - 1), Height / (Ny - 1), Depth / (Nz - 1))
    shape = [Nx, Ny, Nz]
    dxdydz = [hx, hy, hz]
    # ------------------------------ data testing -------------------------------------------------------
    num_test_x = Nx
    num_test_y = Ny
    num_test_z = Nz
    # ------------------------------ filename output ----------------------------------------------------
    out_dir = "simu_normalNet_tx_orientation"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print(f"\nDirectory created: {out_dir:s}\n")
    
    # ----------------------------------------------------------------------
    #                   STEP 1: SETUP DOMAIN - COLLECT CLEAN DATABASE
    # ----------------------------------------------------------------------
    dom, boundary_neumann, boundary_dirichlet = des.setup_domain(
        x_min, Length, Nx,
        y_min, Height, Ny,
        z_min, Depth, Nz,
        known_left_ux, known_left_uy, known_left_uz,
        known_right_tx, known_right_ty, known_right_tz,
        bc_right_penalty,
        bc_left_penalty
    )
    x, y, z, datatest = des.get_datatest(
        x_min, Length, num_test_x,
        y_min, Height, num_test_y,
        z_min, Depth, num_test_z
    )

    # ----------------------------------------------------------------------
    #                   STEP 2: SETUP & TRAIN MODEL
    # ----------------------------------------------------------------------
    for phi_ratio in np.linspace(0., 1.0, 5):
    # for health_coefficient in np.linspace(0, 1.0, 6):
        
        theta = 0.5* np.pi * theta_ratio
        phi = 0.5 * np.pi * phi_ratio
    
        filename_out = f"{out_dir:s}/{model_energy_type:s}_{integration_method:s}_{int(Nx):d}x{int(Ny):d}x{int(Nz):d}_" + \
            f"theta{theta_ratio:.1f}_phi{phi_ratio:.2f}_K{K_penalty:.1e}_mu{mu:.1e}"+\
            f"_layers{num_layers:d}_leftPenalty{bc_left_penalty:.0e}_helth{health_coefficient:.1f}_tx{known_right_tx:.1e}"
        if not os.path.exists(filename_out):
            os.mkdir(filename_out)
            print(f"\nDirectory created: {filename_out:s}\n")
        print("#"*80)
        print(filename_out)
        print("#"*80)
        start_time = time.time()
        f_outstream = open(os.path.join(filename_out, "training_history.txt"), 'w')
        
        # finished: check if the random field is correctly applied
        mat = md.EnergyModel(
            device=device,
            energy=model_energy_type, dim=3, numg=numg,
            K_penalty=K_penalty, mu=mu, c1=c1, c2=c2, phi=phi, theta=theta, 
            healthy_ratio=health_coefficient,
            random_health_flag=False, )
        dem = DeepEnergyMethod([D_in, H, D_out], integration_method, mat, 3, device = device)
        dem.train_model(shape, dxdydz, dom, boundary_neumann, boundary_dirichlet, iteration, lr)
        
        """
            使用Neohookin和MooneyRivlin模型, 与使用Degraded模型的应力计算方法不同
        """
        
        



