
# ------------------------------ network settings ---------------------------------------------------
iteration = 50
lr = 0.5
D_in = 3
H = 30
D_out = 3
# ------------------------------ material parameter -------------------------------------------------
E = 1000
nu = 0.3
param_c1 = 630
param_c2 = -1.2
param_c = 10000
model_energy = 'degraded'
# 肌肉纤维方向 
theta = 0.
phi = 0.
# ----------------------------- define structural parameters ---------------------------------------
Length = 1.0
Height = 1.0
Depth = 0.2
known_left_ux = 0
known_left_uy = 0
known_left_uz = 0
bc_left_penalty = 1.0

known_right_tx = 0
known_right_ty = -5.0
known_right_tz = 0
bc_right_penalty = 1.0
# ------------------------------ define domain and collocation points -------------------------------
Nx = 20  # 120  # 120
Ny = 20  # 30  # 60
Nz = 4   # 30  # 10
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
# filename_out = "./NeoHook3D_beam20x5_NeoHook_traction-1p25_20iter_100_25_5P_pen100000"