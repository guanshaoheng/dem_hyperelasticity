import numpy as np
import torch
from torch_Lagrange.panel_utils import read_gmsh3d, get_random_field, \
    get_triangle_areas_on_right_boundary
from constitutive.cons import DegradedCons, DegradedCons_different_health
import os
import time
from matplotlib import pyplot as plt
import meshio


def cal_von_miese_stress_3d(sig):
    """
        sig: in shape of (num_samples, 3, 3)
    """
    return np.sqrt(
        0.5 * ( (sig[:, 0, 0] - sig[:, 1, 1]) **2 +  (sig[:, 2, 2] - sig[:, 1, 1]) **2 + (sig[:, 0, 0] - sig[:, 2, 2]) **2) +
        3. * (sig[:, 0, 1] ** 2 + sig[:, 1, 2]**2 + sig[:, 0, 2]**2))


class Torch_Lagrange:
    def __init__(
            self, 
            device: torch.device, 
            m:float,
            g:float,
            dh:float,
            theta: float,
            phi: float,
            K_penalty: float,
            mu: float,
            c1: float, 
            c2: float,
            health_coefficient: float, 
            filename_out: str,
            random_index:int=None,) -> None:
        
        # compuational settings
        self.device = device
        self.filename_out = filename_out
        
        # 初始化位置（x）和速度（v）张量
        x, tetra, self.mesh = read_gmsh3d()
        self.x = torch.tensor(x, dtype=torch.float32, device=self.device)
        self.tetra = tetra
        self.v = torch.zeros_like(self.x, device=self.device)
        self.acc = torch.zeros_like(self.x, device=self.device)
        self.n_node = len(self.x)
        self.n_tetra = len(self.tetra)
        if random_index is not None:
            self.random_field_health = torch.tensor(
                get_random_field(x, tetra, random_index), 
                device=self.device)
        else:
            self.random_field_health = None
        """
        Check the random field, if it is working or not
        self.random_field_health = torch.ones_like(self.random_field_health) * random_index
        """
        self.dim = len(self.x[0])

        self.lx = np.max(x[:, 0]) - np.min(x[:, 0])
        self.ly = np.max(x[:, 1]) - np.min(x[:, 1])
        self.lz = np.max(x[:, 2]) - np.min(x[:, 2])
        
        self.elements = torch.zeros(size=[self.n_tetra,self.dim, self.dim], device=self.device)
        self.elements_Dm_inv = torch.zeros_like(self.elements, device=self.device)
        self.F = torch.zeros_like(self.elements, device=self.device)
        self.element_v0 = torch.zeros(size=[self.n_tetra], device=self.device)
        self.elements_energy_dens = torch.zeros(size=[self.n_tetra], device=self.device)
        self.sig = torch.zeros_like(self.elements, device=self.device)
        self.pk1 = torch.zeros_like(self.elements, device=self.device)
        self.pk2 = torch.zeros_like(self.elements, device=self.device)

        # 节点质量，重力向量
        self.m = m
        self.g = torch.tensor([0., 0., -g], dtype=torch.float32, device=self.device)
        self.dh = dh
        
        # material constitutive model
        if random_index is not None:
            self.cons = DegradedCons_different_health(
                healthy_ratio_tensor=self.random_field_health,
                theta=theta, phi=phi, 
                K=K_penalty, mu=mu, c1=c1, c2=c2, healthy_ratio=health_coefficient, 
                device=self.device)
        else:
            self.cons = DegradedCons(
                theta=theta, phi=phi, 
                K=K_penalty, mu=mu, c1=c1, c2=c2, healthy_ratio=health_coefficient, 
                device=self.device,
            )
        # self.cons = NeoHooking(
        #     k=K_penalty, mu=mu) # for debuging

        # 边界约束的索引 左边界固定
        self.boundary_index_left = self.boundary_index_left_func()
        self.boundary_index_right = self.boundary_index_right_func()  # index of points on the right boundary
        self.boundary_index_bottom = self.boundary_index_bottom_func()
        self.boundary_index_top = self.boundary_index_top_func()
        
        # get the triangel facets on the right boundary
        self.right_boundary_triangles_area, self.right_boundary_triangles_node_index = \
            get_triangle_areas_on_right_boundary(self.mesh)

        self.x_origin = self.x.clone()
        self.x_origin_arr = x
        self.init_elements()

        # 输出文件时将cell data 投影到 point data
        self.num_emergences_averaged = self.cal_num_emergences()

        # output simulation log into a file
        self.foutstream = open(os.path.join(f"{filename_out}/loading_log.txt"), 'w')
        line = f"K: {K_penalty:.2e} mu: {mu:.2e} healthy: {health_coefficient:.2e} "\
            f"theta: {theta:.2e} phi: {phi:.2e} \n"\
            f"num_points:{len(self.x)} num_tetra:{len(self.tetra)}"
        self.foutstream.writelines(line + "\n")

        # write the intial configuration into a vtk file
        self.write_to_vtk(epoch=-1)

    def boundary_index_left_func(self,):
        index = torch.arange(self.n_node, device=self.device)[
            torch.isclose(torch.zeros(1), self.x[:, 0])]
        return index
    
    def boundary_index_right_func(self,):
        index = torch.arange(self.n_node, device=self.device)[
            torch.isclose(torch.ones(1)*self.lx, self.x[:, 0])]
        return index
    
    def boundary_index_bottom_func(self, ):
        index = torch.arange(self.n_node, device=self.device)[
            torch.isclose(torch.min(self.x[:, 2]), self.x[:, 2])]
        return index
    
    def boundary_index_top_func(self, ):
        index = torch.arange(self.n_node, device=self.device)[
            torch.isclose(torch.max(self.x[:, 2]), self.x[:, 2])]
        return index
    
    def compute_D(self, x):
        elements = torch.stack([
            x[self.tetra[:, 1]] - x[self.tetra[:, 0]],
            x[self.tetra[:, 2]] - x[self.tetra[:, 0]],
            x[self.tetra[:, 3]] - x[self.tetra[:, 0]],
            ]).permute(1, 2, 0)
        return elements

    def init_elements(self, ):
        self.elements = self.compute_D(self.x)
        self.elements_Dm_inv = torch.linalg.inv(self.elements)
        self.element_v0 = torch.abs(torch.linalg.det(self.elements))/6.
        self.F = self.cal_F(self.elements)
        self.F.requires_grad = True
        self.sig, self.elements_energy_dens, self.pk1, self.pk2 = self.cal_sigma(self.F)

    def cal_F(self, D):
        F = torch.einsum("nij, njk->nik", D, self.elements_Dm_inv)
        return F
    
    def cal_sigma(self, F):
        """
            势能方程参考: https://en.wikipedia.org/wiki/Neo-Hookean_solid
        """
        # F.requires_grad= True
        energy_density, sig, pk1, pk2 = self.cons.get_cauchy_stress_batch(F)
        return sig, energy_density * self.element_v0, pk1, pk2
    
    def train(
            self, 
            iteration: int=int(1e6),
            load_step_len:int = 1000, 
            save_per_epoch: int = 200,
            loss_tolerance: float = 1e-3,
            patience_num: int = 10,
            known_right_epsx: float = 0., 
            known_right_epsy: float = 0.,
            known_right_epsz: float = 0.,
            known_right_tx: float = 0.,
            known_right_ty: float = 0.,
            known_right_tz: float = 0.,
            ):
        # decide what kind of boundary is using here  
        # boundary_type: 1 -> Dirichlet (first, essential)
        # boundary_type: 2 -> Neumann (second)
        if (known_right_epsx!=0. or known_right_epsy!=0 or known_right_epsz!=0):
            boundary_type: int = 1
        elif (known_right_tx != 0. or known_right_ty!=0. or known_right_tz!=0.):
            boundary_type: int = 2
            right_boundary_node_force = self.get_right_boundary_node_force(
                known_right_tx=known_right_tx, 
                known_right_ty=known_right_ty,
                known_right_tz = known_right_tz
            )
        else:
            raise RuntimeError("Please set the boundary conditions!")

        t = 0
        converge_flag = False
        last_loss = 1e5
        try_num = 0
        it_time = time.time()
        while t < iteration and not converge_flag:

            # apply the traction linearly within steps of load_step_len
            factor = min(t, load_step_len) / load_step_len

            # clone the coordinates to 
            x_clone = self.x.clone().detach().requires_grad_(True)
            elements = self.compute_D(x_clone)
            F = self.cal_F(elements)
            energy_elements = self.cons.total_energy_batch(F) * self.element_v0

            # external work if there are Neumann boundary conditions
            if boundary_type==2:
                External = torch.einsum(
                    "nj, nj->", 
                    x_clone[self.boundary_index_right] - self.x_origin[self.boundary_index_right], 
                    right_boundary_node_force * factor)
            else:
                External = 0.

            Internal = torch.sum(energy_elements)
            # self.optimizer.zero_grad()

            loss = Internal - External
            loss.backward()
            self.acc = -x_clone.grad/self.m 
            self.v += self.acc * 0.5* self.dh if t ==0 else self.acc * self.dh

            # set the node velocities to 0 according to the boundary constraints
            self.v[self.boundary_index_left, :] *= 0.
            if boundary_type == 1:
                self.v[self.boundary_index_right, 2] *= 0.
                
            # update the coordinates and apply the first boundary condtions
            self.x += self.dh * self.v
            if boundary_type == 1:
                self.x[self.boundary_index_right, 0] = self.x_origin[self.boundary_index_right, 0] +  \
                    factor * known_right_epsx * self.lx

            # damping
            self.v *= np.exp(-self.dh * 12.)

            if (t+1) % save_per_epoch==0 or t==0:
                if loss.item() < last_loss:
                    improvement = (last_loss- loss.item()) / abs(last_loss)
                else:
                    improvement = -1.
                if improvement < loss_tolerance  and t > save_per_epoch and t>load_step_len:
                    try_num += 1
                    if try_num > patience_num:
                        converge_flag = True
                        print("="*60)
                        print(f"Computation converged with loss "
                            f"{abs((loss.item() - last_loss) / last_loss):.3e} !") 
                else:
                    try_num = 0
                        
                line = 'Iter: %d Loss: %.9e Internal: %.9e External: %.9e Improvement: %.3e Time: %.3e(s) trial: %d/%d' \
                        % (t+1 if t>0 else t, loss.item(), Internal.item(), External, improvement,
                        (time.time() - it_time), try_num, patience_num)
                self.sig, self.elements_energy_dens, self.pk1, self.pk2 = self.cal_sigma(F)
                self.F = F
                print(line)
                self.foutstream.writelines(line+ "\n")

                # write the solutions to vtk files
                if (t+1) % (10 * save_per_epoch) ==0 or t==0:
                    self.write_to_vtk(epoch=t)
                last_loss = loss.item()
            t+=1
        
        right_force = self.cal_force_on_right_boundary()
        line = f"The integrated force on right facet: [ {right_force[0]} {right_force[1]} {right_force[2]} ] "\
            f"consumed_time: {(time.time() - it_time)/60.}(mins)"
        print(line)
        self.foutstream.writelines(line + "\n")
        self.write_to_vtk(epoch=t)
        # self.plot()

    def plot(self):
        """

        """
        tmp = self.x.detach().numpy()
        plt.scatter(tmp[:, 0], tmp[:, 1])
        plt.axis("equal")
        plt.show()
        plt.close()
        
    def write_to_vtk(self, epoch:int):

        x_arr = self.x.cpu().detach().numpy()
        
        sig_node = self.cell_data_to_point(self.sig)
        pk1_node = self.cell_data_to_point(self.pk1)
        pk2_node = self.cell_data_to_point(self.pk2)
        # eps_node = self.cell_data_to_point(
        #     0.5 * (torch.einsum("nji, njk->nik", self.F, self.F) - torch.eye(self.dim))
        #     )
        # infinitesimal strain assumption
        eps_node = self.cell_data_to_point(
            0.5 * (self.F + torch.einsum("nij->nji", self.F)) - torch.eye(self.dim)
            )
        
        mesh = meshio.Mesh(
            points=x_arr,
            cells={"tetra": self.tetra})
            
        mesh.point_data["U"] = x_arr - self.x_origin_arr
        mesh.point_data["Sig_v"] = np.einsum("nii->n", sig_node)/3. 
        mesh.point_data["Sig_mises"] = cal_von_miese_stress_3d(sig=sig_node)
        mesh.point_data["Sig"] = sig_node.reshape(self.n_node, self.dim * self.dim)
        mesh.point_data["pk1"] = pk1_node.reshape(self.n_node, self.dim * self.dim)
        mesh.point_data["pk2"] = pk2_node.reshape(self.n_node, self.dim * self.dim)
        mesh.point_data["Eps_v"] = np.einsum("nii->n", eps_node)
        mesh.point_data["Eps"] = eps_node.reshape(self.n_node, self.dim*self.dim)

        if self.random_field_health is not None:
            health_node = self.cell_scalar_data_to_point(self.random_field_health)
            mesh.point_data["health"] = health_node

        mesh.write(f'{self.filename_out:s}/output_{epoch+1}.vtk')
    
    def get_right_boundary_node_force(
            self, 
            known_right_tx: float, 
            known_right_ty: float, 
            known_right_tz: float):
        """
        Parameters:
        - known_right_tx: pressure on the right boundary (Pascal)
        """
        t = np.array([known_right_tx, known_right_ty, known_right_tz])
        force = np.einsum("n, j->nj", self.right_boundary_triangles_area, t)
        node_force = np.zeros([self.n_node, 3])
        for i, triangle_index in enumerate(self.right_boundary_triangles_node_index):
            node_force[triangle_index] += force[i]/3.  # there are 3 points of 1 triangle.

        """
        finished: Check if the integrated force are right  # right
        area = np.sum(self.right_boundary_triangles_area) 
        tmp = torch.tensor(node_force, dtype=torch.float, device=self.device)[self.boundary_index_right]
        force_total = torch.sum(tmp, dim=0)
        pressure_total = force_total / aread
        """
        return torch.tensor(node_force, dtype=torch.float, device=self.device)[self.boundary_index_right]
    
    def cal_force_on_right_boundary(
            self, 
            n = np.array([1., 0. ,0.])):
        """
            integrating the stress tensor on the right boundary to get the force on right boundary
        """

        """
        finished: Check the traction on the right boundary after optimisation
        """
        # project stress from integration points to node
        sig_node = self.cell_data_to_point(self.sig)
        # traction
        t = np.einsum("nij, i->nj", sig_node, n)
        # traction on every triangles
        t_mean_triangle = np.mean(t[self.right_boundary_triangles_node_index], axis=1)
        force_on_boundary = np.einsum(
            "nj, n->j", t_mean_triangle, self.right_boundary_triangles_area)
        return force_on_boundary


    def cell_data_to_point(self, cell_data: torch.Tensor):
        """
            Project the data in the elements to the points
        """
        node_data = np.zeros(shape=[self.n_node, self.dim, self.dim], dtype=np.float64)
        
        cell_data = np.float64(cell_data.cpu().detach().numpy())

        for i, node_nums in enumerate(self.tetra):
            node_data[node_nums] += cell_data[i]

        node_data = np.einsum(
            "n, nij->nij", self.num_emergences_averaged, node_data)
        return node_data
    
    def cell_scalar_data_to_point(self, cell_data):
        """
            project cell data to points
        """
        node_data= np.zeros(shape=[self.n_node], dtype=np.float64)
        cell_data = np.float64(cell_data.cpu().detach().numpy())
        for i, node_nums in enumerate(self.tetra):
            node_data[node_nums] += cell_data[i]

        node_data = np.einsum(
            "n, n->n", self.num_emergences_averaged, node_data)
        return node_data
    
    def cal_num_emergences(self, ):
        """
            Counts the number of times each point appears in a cell.

            Used for averaging cell data when projecting to points
        """
        node_emerge_num = np.zeros([self.n_node,], dtype=int)
      
        for i, node_nums in enumerate(self.tetra):
            node_emerge_num[node_nums] += 1
        num_emergences_averaged = 1/node_emerge_num
        return num_emergences_averaged