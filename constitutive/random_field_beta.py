import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
import os


np.random.seed(10000)

mpl.rcParams.update({'font.size': 14})

class randomField:
    """
        Genereate and save the random field of the healthy ratio 
            which follows the beta distribution.
    """
    def __init__(
            self, 
            character_l=0.3, character_var=1.0, lx = 2/100, n_points=51, dim=2, nugget =1e-8, 
            minimum = 0.2,
            ) -> None:
        """
        Parameters:
        - minimum: the minimum value of the health ratio
        """
        self.n_points = n_points
        self.dim = dim
        self.character_l, self.character_var = character_l*lx, character_var
        self.minimum = minimum
        self.x = np.linspace(0, lx, self.n_points)
        self.x_mesh, self.y_mesh = np.meshgrid(self.x, self.x)
        self.xx = np.stack([self.x_mesh, self.y_mesh]).transpose(1, 2, 0).reshape(-1, 2)
        """
            plt.scatter(self.xx[:, 0], self.xx[:, 1]); plt.show(); plt.axis("equal")
        """
        self.k = self.gaussian_kernal(self.xx, self.xx) 
        self.L = np.linalg.cholesky(self.k + np.eye(int(self.n_points**self.dim)) * nugget)
        
    def gaussian_kernal(self, x1, x2):
        """
            X is in shape of [num_samples, 2]
        """
        k = self.character_var**2 * np.exp( - (cdist(x1, x2)/self.character_l)**2 / 2. )
        return k
    
    def draw_rgp(self, ):
        return np.dot(self.L, np.random.randn(self.n_points**self.dim)).reshape(self.n_points, self.n_points)

    def draw_gamma(self, ):
        f1 = self.draw_rgp()**2
        f2 = self.draw_rgp()**2
        return f1 + f2
    
    def draw_beta(self, ):
        g1 = self.draw_gamma()
        g2 = self.draw_gamma()
        return g1 / (g1 + g2)
    
    def prepare_save_random(self, n=100, save_dir="random_field_beta_save"):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        betas = np.array([self.draw_beta() for _ in range(n)]) * (1.0 - self.minimum) + self.minimum
        
        for i in range(10):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            surf = plt.contourf(self.x_mesh, self.y_mesh, betas[i], cmap='viridis')
            # surf = ax.plot_surface(self.x_mesh, self.y_mesh, betas[i], cmap='viridis')
            ax.set_xlabel('X Axis')
            ax.set_ylabel('Y Axis')
            # ax.set_zlabel('Z Axis')
            # Add a color bar
            fig.colorbar(surf)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"beta_{i:d}.png"), dpi=200)
            plt.close()
        np.save(os.path.join(save_dir, f"random_beta_num{n:d}.npy"), betas)
        return 


if __name__ == "__main__":
    obj = randomField()
    obj.prepare_save_random()

        