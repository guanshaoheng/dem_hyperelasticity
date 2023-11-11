# ------------------------------ network settings ---------------------------------------------------
iteration = 50
lr = 0.5
D_in = 3
H = 30
D_out = 3
# ------------------------------ material parameter -------------------------------------------------
K_penalty = 8e5
E = 1000
nu = 0.3
param_c1 = 630
param_c2 = -1.2
param_c = 10000
model_energy = 'degraded' # neohookean  degraded
# 肌肉纤维方向 
theta = 0.
phi = 0.
helth_coefficient = 0.  # 1.0 表示完全健康，0.0表示完全损坏
integration_method = "simpson"  # simpson trapezoidal 积分计算方法 
# ----------------------------- define structural parameters ---------------------------------------
Length = 1.0
Height = 1.0
Depth = 0.2
known_left_ux = 0
known_left_uy = 0
known_left_uz = 0
bc_left_penalty = 1.0

known_right_tx = 0
known_right_ty = 0
known_right_tz = -500.
bc_right_penalty = 1.0
# ------------------------------ define domain and collocation points -------------------------------
Nx = int(40/2)  # 120  # 120
Ny = int(40/2)  # 30  # 60
Nz = int(8/2)   # 30  # 10
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
filename_out = f"./{model_energy:s}_{integration_method:s}_beam{int(Nx):d}x{int(Ny):d}x{int(Nz):d}_" + \
    f"theta{int(theta):d}_phi{int(phi):d}_helth{helth_coefficient:.1f}_iter{iteration:d}" 

# --------------------------------------------------ECHO---------------------------------------------
print("#"*80)
print(filename_out)
print("#"*80)