import taichi as ti
from pyevtk.hl import gridToVTK
import numpy as np


ti.init(arch=ti.cpu)  # 或 ti.gpu

# 定义 Taichi 字段
N = 32
field = ti.field(dtype=ti.f32, shape=(N, N, N))
output_interval  = 1
num_steps = 10



@ti.kernel
def update_field(t:int):
    for i, j, k in field:
        field[i, j, k] = ti.sin(i * 0.1 * np.pi * t/10) * \
            ti.cos(j * 0.1 * np.pi * t/10) * \
            ti.sin(k * 0.1 * np.pi * t/10)


def write_to_vtu(step):
    # 将 Taichi 字段转换为 NumPy 数组
    data_np = field.to_numpy()

    # 创建网格坐标
    x = np.arange(0, N)
    y = np.arange(0, N)
    z = np.arange(0, N)

    xx, yy, zz = np.meshgrid(x, y, z)

    # 输出 VTU 文件
    gridToVTK(f'output_{step}', xx, yy, zz, pointData={'field': data_np})

# 主循环
for step in range(num_steps):  # num_steps 是你的模拟步数
    update_field(step)
    if step % output_interval == 0:  # output_interval 是输出间隔
        write_to_vtu(step)
