import torch
import numpy as np
import meshio


def rotation_matrix(theta: torch.tensor, phi: torch.tensor, device=torch.device("cpu"))-> torch.tensor:
    r"""
    从原始坐标系，将z轴 绕轴 k=[nx, ny, nz] 旋转角度 theta （右手法则）
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    """
    z_ = torch.tensor([0., 0., 1.0], device=device)
    n_xy = torch.tensor([torch.cos(phi), torch.sin(phi), 0.])
    n = torch.cross(z_, n_xy)  # the vector about which the z axis is rotated
    nx, ny, nz = n / torch.norm(n)

    K = torch.tensor([
        [0., -nz, ny],
        [nz, 0., -nx],
        [-ny, nx, 0.]
    ])

    R = torch.eye(3, device=device) + torch.sin(theta) * K + (1- torch.cos(theta)) * K @ K

    # z_rotated = R@z_
    # n = torch.tensor([sin_theta*torch.cos(phi), sin_theta*torch.sin(phi), cos_theta])
    return R


def read_integration_sphere():
    """ 
        a half sphere with a radius of 1.0
        areas:   in shape of [num_triangles,]
        centres: in shape of [num_triangles, 3]
    """
    mesh = meshio.read(
        "/home/shguan/dem_hyperelasticity/constitutive"
        "/mesh_sphere_integration/half_spherical_mesh.msh")
    
    # extract nodes
    points = np.array(mesh.points)
    # extract the triangels on the spherical surface
    triangles = np.array([cell.data for cell in mesh.cells if cell.type == "triangle"][0], dtype=int)

    areas = []
    centres = []
    for coord_index in triangles:
        a, b, c = points[coord_index]
        centre = (a+b+c)/3.
        centre_len = np.linalg.norm(centre)
        centres.append(centre/centre_len*1.)
        areas.append(triangle_area(a, b, c))
    areas = np.array(areas)
    centres = np.array(centres)[:, [0, 2, 1]]

    print(f"\nNumber of the integration triangles: {len(centres)}\n")

    weights = weight_on_spere(cos_theta=centres[:, 2])
    weights_normed = weights/np.sum(weights)
    
    mesh_new = meshio.Mesh(
        points=points, 
        cells={"triangle": triangles},
        cell_data={"weights":[weights_normed]},
        )
    
    mesh_new.write(
        "/home/shguan/dem_hyperelasticity/constitutive"
        "/mesh_sphere_integration/sphere_surface_integration.vtk")
    
    areas = torch.tensor(areas).float()
    centres = torch.tensor(centres).float()

    return areas, centres

# Function to calculate the area of a triangle
def triangle_area(a, b, c):
    # a = np.array([0, 0, 0]); b = np.array([0, 1, 0]); c = np.array([1, 0, 0])

    return 0.5 * np.linalg.norm(np.cross(b-a, c-a))


def weight_on_spere(cos_theta, be=1.0, ):
    return np.exp(be * cos_theta**2)


if __name__ == "__main__":
    _  = read_integration_sphere()

