import numpy as np
import os

# ------------------------------ network settings ---------------------------------------------------
iteration = int(1e6)
lr = 0.5
D_in = 3
H = 30
D_out = 3
save_per_epoch = 1e3
internal_tolerance = 0.001
patience_num = 5
# ------------------------------ material parameter -------------------------------------------------

E = 1e6
nu = 0.2
lameLa = E*nu / ((1+nu)*(1-2*nu))
lameMu = E / (2*(1+nu))
dh = 16.7e-5*2.
m = 1.0
g= 9.8

# 肌肉纤维方向 
theta = 0.
phi = 0.
helth_coefficient = 0.  # 1.0 表示完全健康，0.0表示完全损坏
integration_method = "simpson"  # simpson trapezoidal 积分计算方法 
# ----------------------------- define structural parameters ---------------------------------------
lx = Length = 1.0
ly = Height = 1.0
lz = Depth = 0.2
init_x, init_y = 0.1, 0.6

known_left_ux = 0
known_left_uy = 0
known_left_uz = 0
bc_left_penalty = 1.0

known_right_ux = 5.0
known_right_uy = 0.0
known_right_uz = 0.

known_right_tx = 0.
known_right_ty = 0.
known_right_tz = 0.
bc_right_penalty = 1.0

# 将位移在1000个荷载步上逐渐施加上去
load_step_len = 1000

# ------------------------------ define domain and collocation points -------------------------------
nx = Nx = int(50)  # 120  # 120
ny = Ny = int(10)  # 30  # 60
nz = Nz = int(1)   # 30  # 10
dim = 2
n_node = Nx*Ny*Nz
x_min, y_min, z_min = (0.0, 0.0, 0.0)
dx = dy = dz = 1/32
shape = [Nx, Ny, Nz]
dxdydz = [dx, dy, dz]
n_triangles = (nx-1) * (ny-1)*2

xx, yy, zz = np.meshgrid(
    np.linspace(init_x, init_x + dx * (nx - 1), nx), 
    np.linspace(init_y, init_y + dx * (ny - 1), ny),
    np.linspace(0, 1e-6, 2)
    )
# ------------------------------ data testing -------------------------------------------------------
nx_test = int(10)
ny_test = int(10)
nz_test = int(20)
hx_test = lx/(nx_test-1)
hy_test = ly/(ny_test-1)
hz_test = lz/(nz_test-1)

# ------------------------------ filename output ----------------------------------------------------
filename_out = f"./Tshape_2D_{int(Nx):d}x{int(Ny):d}x{int(Nz):d}_" + \
    f"E{E:.0e}_nu{nu:.0e}" 

if not os.path.exists(filename_out):
    os.mkdir(filename_out)

# --------------------------------------------------ECHO---------------------------------------------
print("#"*80)
print(filename_out)
print("#"*80)