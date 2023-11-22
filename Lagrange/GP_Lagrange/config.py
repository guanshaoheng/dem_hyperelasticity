# ------------------------------ network settings ---------------------------------------------------
iteration = 300
lr = 0.5
D_in = 3
H = 30
D_out = 3
# ------------------------------ material parameter -------------------------------------------------
E = 1e3
nu = 0.2
K_penalty = 8e10
mu = 62.1e3
c1 = 56.59e3
c2 = 3.83
model_energy = 'degraded' # neohookean  degraded
# 肌肉纤维方向 
theta = 0.
phi = 0.
helth_coefficient = 0.  # 1.0 表示完全健康，0.0表示完全损坏
integration_method = "simpson"  # simpson trapezoidal 积分计算方法 
# ----------------------------- define structural parameters ---------------------------------------
lx = Length = 1.0
ly = Height = 1.0
lz = Depth = 0.2
known_left_ux = 0
known_left_uy = 0
known_left_uz = 0
bc_left_penalty = 1.0

known_right_tx = 5000.
known_right_ty = 0.
known_right_tz = 0.
bc_right_penalty = 1.0
# ------------------------------ define domain and collocation points -------------------------------
Nx = int(30)  # 120  # 120
Ny = int(30)  # 30  # 60
Nz = int(10)   # 30  # 10
numg = Nx*Ny*Nz
x_min, y_min, z_min = (0.0, 0.0, 0.0)
(hx, hy, hz) = (Length / (Nx - 1), Height / (Ny - 1), Depth / (Nz - 1))
shape = [Nx, Ny, Nz]
dxdydz = [hx, hy, hz]
# ------------------------------ data testing -------------------------------------------------------
nx =  Nx
ny =  Ny
nz =  Nz
nx_test = int(10)
ny_test = int(10)
nz_test = int(20)
hx_test = lx/(nx_test-1)
hy_test = ly/(ny_test-1)
hz_test = lz/(nz_test-1)

# ------------------------------ filename output ----------------------------------------------------
filename_out = f"./{model_energy:s}_{integration_method:s}_beam{int(Nx):d}x{int(Ny):d}x{int(Nz):d}_" + \
    f"theta{int(theta):d}_phi{int(phi):d}_helth{helth_coefficient:.1f}_K{K_penalty:.0e}_mu{mu:.0e}_iter{iteration:d}" 

# --------------------------------------------------ECHO---------------------------------------------
print("#"*80)
print(filename_out)
print("#"*80)