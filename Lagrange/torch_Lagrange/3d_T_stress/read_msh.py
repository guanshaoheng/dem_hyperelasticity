import meshio
import numpy as np
import matplotlib.pyplot as plt

def read_gmsh3d(fname = "./mesh_files/T_shape_structure.msh"):
    # 读取.msh文件
    mesh = meshio.read(fname)

    # 提取节点
    points = np.array(mesh.points)
    # 提取三角形单元
    # 注意：这里假设你的网格是由三角形单元构成的
    tetra = np.array([cell.data for cell in mesh.cells if cell.type == "tetra"], dtype=np.int)
    
    mesh_new = meshio.Mesh(points= points, cells={"tetra": tetra[0]})

    mesh_new.write(f"./mesh_files/Original shape.vtk")

    print(f"\n\nNum_elements: {len(points):d} Num_tetra: {len(tetra[0]):d}\n\n")
    
    return points, tetra[0]


if __name__ == "__main__":
    read_gmsh3d()