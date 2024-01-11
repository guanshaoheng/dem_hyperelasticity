import numpy as np
import torch
from pyevtk.hl import gridToVTK, pointsToVTK



def get_random_field(
        coords: np.ndarray,
        random_index: int):
    """
    Read the random field, and interpolate to get the random field with the 
        same shape of the input coords

    Parameters:
    - coords (np.ndarray):  in shape of [num_points, 3]
    - random_index (int): index of the generated random field

    Returns:
    - random_field_interpolated (np.ndarray): in shape of [num_points, ]
    """
    minx =  np.min(coords[:, 0])
    miny =  np.min(coords[:, 1])
    lx = np.array([
        np.max(coords[:, 0]) - minx,
        np.max(coords[:, 1]) - miny,
        np.max(coords[:, 2]) - np.min(coords[:, 2]),
        ])

    random_field = np.load("../constitutive/random_field_beta_save/random_beta_num100.npy")[random_index]

    nx, ny = random_field.shape

    # FINISHED: CHANGE HERE TO THE LINEAR INTERPOLATION INSTEAD TO GET HIGHER ACCURACY
    random_field_interpolated = np.zeros(len(coords))
    for i, coord in enumerate(coords):
        index_x = (coord[0]-minx)/lx[0] * (nx-1.01)
        index_y = (coord[1]-miny)/lx[1] * (ny-1.01)
        res_x, res_y = index_x%1, index_y%1
        index_x, index_y = int(index_x), int(index_y)
        random_field_interpolated[i] = interpolation(
            res_x, res_y, 
            random_field[index_x, index_y], random_field[index_x, index_y+1],
            random_field[index_x+1, index_y], random_field[index_x+1, index_y+1])
    return random_field_interpolated


def interpolation(res_x: float, res_y: float, a:float, b: float, c: float, d: float):
    tmp = (1-res_x)*(1-res_y) * a + \
        (1-res_x) *  res_y * b + \
            res_x * (1-res_y) * c + \
                res_x * res_y * d
    return tmp
    

def write_to_vtk(
    filename, epoch: int,
    x_space, y_space, z_space, 
    U, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises, 
    E11_cauchy, E12_cauchy, E13_cauchy, E22_cauchy, E23_cauchy, E33_cauchy,
    pk1_11, pk1_12 , pk1_13, pk1_21, pk1_22 , pk1_23, pk1_31, pk1_32 , pk1_33, 
    pk2_11, pk2_12 , pk2_13, pk2_21, pk2_22 , pk2_23, pk2_31, pk2_32 , pk2_33,
    random_field: np.ndarray,
    ):
    """
    Parameters:
    - random_field (np.ndarray): [num_points,]
    """
    xx, yy, zz = np.meshgrid(x_space, y_space, z_space)
    gridToVTK(
        filename + f"/deformed_{epoch}", 
        xx + U[0], yy + U[1], zz + U[2],
        pointData={
            "displacement": U, 
            "S-VonMises": SVonMises, 
            # cauchy stress, sigma
            "S11": S11, "S12": S12, "S13": S13, "S22": S22, "S23": S23, "S33": S33, 
            "sig_v": (S11+S22+S33)/3.,
            # strain under large deformation (green strain)
            "E11": E11, "E12": E12, "E13": E13, "E22": E22, "E23": E23, "E33": E33,
            "eps_v": E11+E22+E33,
            # cauchy strain
            "E11cauchy": E11_cauchy, "E12cauchy": E12_cauchy, "E13cauchy": E13_cauchy, 
            "E22cauchy": E22_cauchy, "E23cauchy": E23_cauchy, "E33cauchy": E33_cauchy,
            "eps_v_cauchy": E11_cauchy+E22_cauchy+E33_cauchy,
            # pk1
            "pk1_11": pk1_11, "pk1_12": pk1_12, "pk1_13": pk1_13, 
            "pk1_22": pk1_22, "pk1_23": pk1_23, "pk1_33": pk1_33,
            # pk2
            "pk2_11": pk2_11, "pk2_12": pk2_12, "pk2_13": pk2_13, 
            "pk2_22": pk2_22, "pk2_23": pk2_23, "pk2_33": pk2_33,
            # random field
            "random_field": np.ascontiguousarray(random_field),
            })
    gridToVTK(
        filename + f"/undeformed_{epoch}", 
        xx, yy, zz,
        pointData={
            "displacement": U, "S-VonMises": SVonMises, 
            "S11": S11, "S12": S12, "S13": S13, "S22": S22, "S23": S23, "S33": S33, 
            "E11": E11, "E12": E12, "E13": E13, "E22": E22, "E23": E23, "E33": E33,
            "random_field": np.ascontiguousarray(random_field),
        },
    )