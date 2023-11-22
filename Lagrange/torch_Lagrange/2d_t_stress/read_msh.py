import meshio
import numpy as np
import matplotlib.pyplot as plt

def read_gmsh(fname = "./mesh_files/t_2d.msh"):
    # 读取.msh文件
    mesh = meshio.read(fname)

    # 提取节点
    points = np.array(mesh.points)[:, :2]
    # 提取三角形单元
    # 注意：这里假设你的网格是由三角形单元构成的
    triangles = np.array([cell.data for cell in mesh.cells if cell.type == "triangle"], dtype=np.int)
    
    # plt.close();plt.scatter(points[:, 0], points[:, 1])
    # for i in range(len(points)):
    #     plt.text(points[i, 0], points[i, 1], str(i))
    # plt.axis("equal"); plt.show()

    return points, triangles[0]


if __name__ == "__main__":
    read_gmsh()