import taichi as ti
import math
import numpy as np
from pyevtk.hl import gridToVTK


ti.init(arch=ti.cpu)

# global control
dim = 2
paused = True
damping_toggle = ti.field(ti.i32, ())
damping_toggle[None] = not damping_toggle[None]
curser = ti.Vector.field(2, ti.f32, ())
picking = ti.field(ti.i32,())
using_auto_diff = False

# procedurally setting up the cantilever
init_x, init_y = 0.1, 0.6
N_x = 20
N_y = 4
N = N_x*N_y
N_edges = (N_x-1)*N_y + N_x*(N_y - 1) + (N_x-1) * \
    (N_y-1)  # horizontal + vertical + diagonal springs
N_triangles = 2 * (N_x-1) * (N_y-1)
dx = 1/32
curser_radius = dx/2
xx, yy, zz = np.meshgrid(
    np.linspace(init_x, init_x + dx * (N_x - 1), N_x), 
    np.linspace(init_y, init_y + dx * (N_y - 1), N_y),
    np.linspace(0, 1e-6, 2)
    )

# physical quantities
m = 1
g = 9.8
YoungsModulus = ti.field(ti.f32, ())
PoissonsRatio = ti.field(ti.f32, ())
LameMu = ti.field(ti.f32, ())
LameLa = ti.field(ti.f32, ())

# 施加一个力边界条件在右端
fy = 1.

# time-step size (for simulation, 16.7ms)
h = 16.7e-3
# substepping
substepping = 100
# time-step size (for time integration)
dh = h/substepping

# simulation components
x = ti.Vector.field(2, ti.f32, N, needs_grad=True)
v = ti.Vector.field(2, ti.f32, N)
acc = ti.Vector.field(2, ti.f32, N)
max_acc = ti.field(ti.f32, ())
max_acc_index = ti.field(ti.i32, ())
total_energy = ti.field(ti.f32, (), needs_grad=True)
grad = ti.Vector.field(2, ti.f32, N)
elements_Dm_inv = ti.Matrix.field(2, 2, ti.f32, N_triangles)
F = ti.Matrix.field(2, 2, ti.f32, N_triangles, needs_grad=True)
sig = ti.Matrix.field(2, 2, ti.f32, N_triangles)
elements_energy_dens = ti.field(ti.f32, N_triangles, needs_grad=True)
elements_V0 = ti.field(ti.f32, N_triangles)

# geometric components
triangles = ti.Vector.field(3, ti.i32, N_triangles)
edges = ti.Vector.field(2, ti.i32, N_edges)

@ti.func
def ij_2_index(i, j): 
    return i * N_y + j

# -----------------------meshing and init----------------------------
@ti.kernel
def meshing():
    # setting up triangles
    for i,j in ti.ndrange(N_x - 1, N_y - 1):
        # triangle id
        tid = (i * (N_y - 1) + j) * 2
        triangles[tid][0] = ij_2_index(i, j)
        triangles[tid][1] = ij_2_index(i + 1, j)
        triangles[tid][2] = ij_2_index(i, j + 1)

        tid = (i * (N_y - 1) + j) * 2 + 1
        triangles[tid][0] = ij_2_index(i, j + 1)
        triangles[tid][1] = ij_2_index(i + 1, j + 1)
        triangles[tid][2] = ij_2_index(i + 1, j)

    # setting up edges
    # edge id
    eid_base = 0

    # horizontal edges
    for i in range(N_x-1):
        for j in range(N_y):
            eid = eid_base+i*N_y+j
            edges[eid] = [ij_2_index(i, j), ij_2_index(i+1, j)]

    eid_base += (N_x-1)*N_y
    # vertical edges
    for i in range(N_x):
        for j in range(N_y-1):
            eid = eid_base+i*(N_y-1)+j
            edges[eid] = [ij_2_index(i, j), ij_2_index(i, j+1)]

    eid_base += N_x*(N_y-1)
    # diagonal edges
    for i in range(N_x-1):
        for j in range(N_y-1):
            eid = eid_base+i*(N_y-1)+j
            edges[eid] = [ij_2_index(i+1, j), ij_2_index(i, j+1)]

@ti.kernel
def initialize():
    YoungsModulus[None] = 1e6
    PoissonsRatio[None] = 0.49
    paused = True
    # init position and velocity
    for i, j in ti.ndrange(N_x, N_y):
        index = ij_2_index(i, j)
        x[index] = ti.Vector([init_x + i * dx, init_y + j * dx])
        v[index] = ti.Vector([0.0, 0.0])

@ti.func
def compute_D(i):
    a = triangles[i][0]
    b = triangles[i][1]
    c = triangles[i][2]
    return ti.Matrix.cols([x[b] - x[a], x[c] - x[a]])

@ti.kernel
def initialize_elements():
    for i in range(N_triangles):
        Dm = compute_D(i)
        elements_Dm_inv[i] = Dm.inverse()  # 存储每个单元的 D
        elements_V0[i] = ti.abs(Dm.determinant())/2  # 存储每个三角形单元的体积

# ----------------------core-----------------------------
@ti.func
def compute_R_2D(F):
    R, S = ti.polar_decompose(F, ti.f32)
    return R

@ti.kernel
def compute_gradient():
    # clear gradient
    for i in grad:
        grad[i] = ti.Vector([0, 0])

    # gradient of elastic potential
    for i in range(N_triangles):
        Ds = compute_D(i)
        F = Ds@elements_Dm_inv[i]
        # co-rotated linear elasticity
        R = compute_R_2D(F)
        Eye = ti.Matrix.cols([[1.0, 0.0], [0.0, 1.0]])
        # first Piola-Kirchhoff tensor
        P = 2*LameMu[None]*(F-R) + LameLa[None]*((R.transpose())@F-Eye).trace()*R
        #assemble to gradient
        H = elements_V0[i] * P @ (elements_Dm_inv[i].transpose())
        a,b,c = triangles[i][0],triangles[i][1],triangles[i][2]
        gb = ti.Vector([H[0,0], H[1, 0]])
        gc = ti.Vector([H[0,1], H[1, 1]])
        ga = -gb-gc
        grad[a] += ga
        grad[b] += gb
        grad[c] += gc


energy_single = ti.field(ti.float32, (), needs_grad=True)
@ti.kernel
def compute_total_energy():
    for i in range(N_triangles):
        Ds = compute_D(i)
        F[i] = Ds @ elements_Dm_inv[i]
        with ti.Tape(energy_single):
            J = F[i].determinant()
            C = F[i].transpose() @ F[i]
            I_1 = C.trace()
            energy_single[None] = 0.5 * LameMu[None]* (I_1 - 2 - 2* ti.log(J)) + \
                0.5*LameLa[None] * (J-1)**2
        P = F.grad[i]   # PK1应力
        sig[i] = P @ F[i].transpose() / J
        elements_energy_dens[i] = energy_single[None]
        total_energy[None] += elements_energy_dens[i] * elements_V0[i]

@ti.kernel
def update(t:int):
    # perform time integration
    for i in range(N): # 遍历每个点的位置
        # symplectic integration
        # elastic force + gravitation force, divding mass to get the acceleration
        if using_auto_diff:  # 更新加速度和速度位置
            acc[i] = -x.grad[i]/m - ti.Vector([0.0, g])
            v[i] += dh*acc[i]
        else:
            acc[i] = -grad[i]/m - ti.Vector([0.0, g])
            v[i] += dh*acc[i]
        x[i] += dh*v[i]  # 更新位置

    # explicit damping (ether drag)
    for i in v:
        if damping_toggle[None]:
            v[i] *= ti.exp(-dh * 5 * 10)

    # 施加边界条件
    for i in range(N):  # 鼠标边界条件
        if picking[None]:
            r = x[i]-curser[None]
            if r.norm() < curser_radius:
                x[i] = curser[None]
                v[i] = ti.Vector([0.0, 0.0])
                pass

    # 施加边界条件，保证左边界靠近墙
    for j in range(N_y):
        ind = ij_2_index(0, j)   # 选取左边界的点的索引
        v[ind] = ti.Vector([0, 0])    # 将速度设置为0
        x[ind] = ti.Vector([init_x, init_y + j * dx])  # 设置正确的x的位置

    # 如果有点的位置到达左边界，保证该点不超过左边界
    for i in range(N):
        if x[i][0] < init_x:
            x[i][0] = init_x
            v[i][0] = 0

    # 施加位移边界条件， 右边界向上移动
    # tmp = min(t, 10000)/10000
    # for j in range(N_y):
    #     ind = ij_2_index(N_x-1, j)
    #     v[ind] = ti.Vector([0, 0])
    #     x[ind] = ti.Vector([init_x + (N_x-1) * dx + 0.1 *tmp , init_y + j * dx + 0.05*tmp])  # 设置正确的x的位置


@ti.kernel
def updateLameCoeff():
    E = YoungsModulus[None]
    nu = PoissonsRatio[None]
    LameLa[None] = E*nu / ((1+nu)*(1-2*nu))
    LameMu[None] = E / (2*(1+nu))

@ti.kernel
def cal_max_acc():
    max_acc_tmp = 0.
    for i in acc:
        tmp = ti.sqrt(acc[i][0]**2 + acc[i][1]**2)
        if tmp>max_acc_tmp:
            max_acc_tmp =  tmp
            max_acc_index[None] = i
    max_acc[None] = max_acc_tmp


def write_to_vtk(epoch:int):
    x_arr = np.float64(x.to_numpy().reshape(N_x, N_y, dim))
    sig_arr = np.float64(sig.to_numpy())
    sig_arr = np.array([(sig_arr[2*i] + sig_arr[2*i+1])*0.5 for i in range(0, (N_x-1)*(N_y-1))]).reshape(N_x-1, N_y-1, dim, dim)

    x_arr_x = np.ascontiguousarray(np.concatenate((x_arr[..., 0:1], x_arr[..., 0:1]), axis=2).transpose(1, 0, 2))
    x_arr_y = np.ascontiguousarray(np.concatenate((x_arr[..., 1:2], x_arr[..., 1:2]), axis=2).transpose(1, 0, 2))

    gridToVTK(f'output_{epoch}', 
                x_arr_x, x_arr_y, zz, 
                pointData={'ux': np.ascontiguousarray((x_arr_x - xx)), 
                           'uy': np.ascontiguousarray((x_arr_y - yy))},
                cellData={
                    "sig_xx": np.ascontiguousarray(sig_arr[..., 0, 0].transpose(1, 0)[:, :, np.newaxis]),
                    "sig_xy": np.ascontiguousarray(sig_arr[..., 0, 1].transpose(1, 0)[:, :, np.newaxis]),
                    "sig_yx": np.ascontiguousarray(sig_arr[..., 1, 0].transpose(1, 0)[:, :, np.newaxis]),
                    "sig_yy": np.ascontiguousarray(sig_arr[..., 1, 1].transpose(1, 0)[:, :, np.newaxis])}
                )


# init once and for all
meshing()
initialize()
initialize_elements()
updateLameCoeff()

epoch = 0
gui = ti.GUI('Linear FEM', (800, 800))
while gui.running:

    picking[None]=0
    # 运行代码
    for i in range(substepping):
        if using_auto_diff:
            total_energy[None]=0
            with ti.Tape(total_energy):
                compute_total_energy()
        else:
            compute_gradient()
        update(t=epoch*substepping + i)
    
    # print(sig[i].to_numpy())
    
    # 检查加速度大小
    cal_max_acc()
    line = f"Step {epoch} Energy {total_energy[None]:.2e} max_acc {max_acc[None]:.2e}"
    print(line)

    # render
    pos = x.to_numpy()
    for i in range(N_edges):
        a, b = edges[i][0], edges[i][1]
        gui.line((pos[a][0], pos[a][1]),
                 (pos[b][0], pos[b][1]),
                 radius=1,
                 color=0xFFFF00)
    gui.line((init_x, 0.0), (init_x, 1.0), color=0xFFFFFF, radius=4)
    # 在图中显示最大加速度的点
    gui.circle((pos[max_acc_index[None]][0], pos[max_acc_index[None]][1]), radius=curser_radius*800, color=0xFF8888)


    # text
    gui.text(
        content=f'9/0: (-/+) Young\'s Modulus {YoungsModulus[None]:.1f}', pos=(0.6, 0.9), color=0xFFFFFF)
    gui.text(
        content=f'7/8: (-/+) Poisson\'s Ratio {PoissonsRatio[None]:.3f}', pos=(0.6, 0.875), color=0xFFFFFF)
    if damping_toggle[None]:
        gui.text(
            content='D: Damping On', pos=(0.6, 0.85), color=0xFFFFFF)
    else:
        gui.text(
            content='D: Damping Off', pos=(0.6, 0.85), color=0xFFFFFF)
    gui.show()

    if epoch % 10==0:
        write_to_vtk(epoch=epoch)

    epoch += 1