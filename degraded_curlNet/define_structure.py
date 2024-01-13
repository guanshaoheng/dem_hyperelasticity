from import_file import *
from config import *


def setup_domain(plot_flag = False):
    x_dom = x_min, Length, Nx
    y_dom = y_min, Height, Ny
    z_dom = z_min, Depth, Nz
    # create points
    lin_x = np.linspace(x_dom[0], x_dom[1], x_dom[2])
    lin_y = np.linspace(y_dom[0], y_dom[1], y_dom[2])
    lin_z = np.linspace(z_dom[0], z_dom[1], z_dom[2])
    dom = np.zeros((Nx * Ny * Nz, 3))
    c = 0
    for z in np.nditer(lin_z):
        for x in np.nditer(lin_x):
            tb = y_dom[2] * c
            te = tb + y_dom[2]
            c += 1
            dom[tb:te, 0] = x
            dom[tb:te, 1] = lin_y
            dom[tb:te, 2] = z
    # print(dom.shape)
    np.meshgrid(lin_x, lin_y, lin_z)

    # ------------------------------------ BOUNDARY ----------------------------------------
    # Left boundary condition (Dirichlet BC)
    bcl_u_pts_idx = np.where(dom[:, 0] == x_min)[0]  # 选中最左侧点的编号
    bcl_u_pts = dom[bcl_u_pts_idx, :]                # 最左侧点
    bcl_u = np.ones(np.shape(bcl_u_pts)) * [known_left_ux, known_left_uy, known_left_uz]

    # Right boundary condition (Neumann BC)
    bcr_t_pts_idx = np.where(dom[:, 0] == Length)[0]
    bcr_t_pts = dom[bcr_t_pts_idx, :]
    bcr_t = np.ones(np.shape(bcr_t_pts)) * [known_right_tx, known_right_ty, known_right_tz]

    # ------------------------------------   PLOT   ----------------------------------------
    if plot_flag:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(dom[:, 0], dom[:, 1], dom[:, 2], s=0.005, facecolor='blue')
        ax.set_xlabel('X', fontsize=3)
        ax.set_ylabel('Y', fontsize=3)
        ax.set_zlabel('Z', fontsize=3)
        ax.tick_params(labelsize=4)
        ax.view_init(elev=15., azim=-85)
        ax.scatter(dom[:, 0], dom[:, 1], dom[:, 2], s=0.005, facecolor='blue')
        ax.scatter(bcl_u_pts[:, 0], bcl_u_pts[:, 1], bcl_u_pts[:, 2], s=0.5, facecolor='red')
        ax.scatter(bcr_t_pts[:, 0], bcr_t_pts[:, 1], bcr_t_pts[:, 2], s=0.5, facecolor='green')
        ax.set_aspect('equal')
        plt.show()

    boundary_neumann = {
        # condition on the right
        "neumann_1": {
            "coord": bcr_t_pts,
            "known_value": bcr_t,
            "penalty": bc_right_penalty
        }
        # adding more boundary condition here ...
    }
    boundary_dirichlet = {
        # condition on the left
        "dirichlet_1": {
            "coord": bcl_u_pts,
            "known_value": bcl_u,
            "penalty": bc_left_penalty
        }
        # adding more boundary condition here ...
    }
    return dom, boundary_neumann, boundary_dirichlet


# -----------------------------------------------------------------------------------------------------
# prepare inputs for testing the model
# -----------------------------------------------------------------------------------------------------
def get_datatest():
    x_dom_test = x_min, Length, num_test_x
    y_dom_test = y_min, Height, num_test_y
    z_dom_test = z_min, Depth, num_test_z
    # create points
    x_space = np.linspace(x_dom_test[0], x_dom_test[1], x_dom_test[2])
    y_space = np.linspace(y_dom_test[0], y_dom_test[1], y_dom_test[2])
    z_space = np.linspace(z_dom_test[0], z_dom_test[1], z_dom_test[2])
    yGrid, xGrid, zGrid = np.meshgrid(x_space, y_space, z_space)
    data_test = np.concatenate(
        (np.array([xGrid.flatten()]).T, np.array([yGrid.flatten()]).T, np.array([zGrid.flatten()]).T), axis=1)
    return x_space, y_space, z_space, data_test


if __name__ == '__main__':
    setup_domain()