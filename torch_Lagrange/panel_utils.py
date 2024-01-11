import meshio
import numpy as np
import matplotlib.pyplot as plt
import torch

def read_gmsh3d(fname = "../mesh_files/panel_3d.msh"):
    # 读取.msh文件
    mesh = meshio.read(fname)

    # 提取节点
    points = np.array(mesh.points)
    # 提取三角形单元
    # 注意：这里假设你的网格是由三角形单元构成的
    tetra = np.array([cell.data for cell in mesh.cells if cell.type == "tetra"], dtype=int)
    
    mesh_new = meshio.Mesh(points= points, cells={"tetra": tetra[0]})

    mesh_new.write(f"../mesh_files/Original_shape.vtk")

    print(f"\nNum_elements: {len(points):d} Num_tetra: {len(tetra[0]):d}\n")
    
    return points, tetra[0], mesh

def get_triangle_areas_on_right_boundary(mesh, tolerance=1e-6):
    max_x = np.max(mesh.points[:, 0])
    # Find tetrahedra on the right boundary and calculate the area of boundary triangles
    boundary_triangle_areas = []
    boundary_triangle_node_index = []

    for cell in mesh.cells:
        if cell.type == "tetra":
            for tetra in cell.data:
                # Get the points of the tetrahedron
                points = mesh.points[tetra]

                # Check each face of the tetrahedron
                for face in [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]:
                    face_points = points[face]
                    
                    # Check if the face is on the right boundary
                    if np.all(np.abs(face_points[:, 0] - max_x) <= tolerance):
                        # Calculate and store the area of the boundary triangle
                        area = triangle_area(face_points)
                        boundary_triangle_areas.append(area)
                        boundary_triangle_node_index.append(tetra[face])
                        # print(f"Triangle {tetra[face]}: {area:.1e}")

    return np.array(boundary_triangle_areas), np.array(boundary_triangle_node_index)
    

# Function to calculate the area of a triangle
def triangle_area(coords):
    a, b, c = coords
    return 0.5 * np.linalg.norm(np.cross(b-a, c-a))

def get_random_field(points, tetras, random_index):
    """
        read the random field, and interpolate to get the random field with the 
            same shape of the integration points
    """
    lx = np.array([
        np.max(points[:, 0]) - np.min(points[:, 0]),
        np.max(points[:, 1]) - np.min(points[:, 1])])

    random_field = np.load("../../constitutive/random_field_beta_save/random_beta_num100.npy")[random_index]

    nx, ny = random_field.shape

    random_field_interpolated = np.zeros(len(tetras))
    for i, tetra in enumerate(tetras):
        tetra_center = np.average(points[tetra, :2], axis=0)/lx
        index_x = min(round(tetra_center[0] * nx), 50)
        index_y = min(round(tetra_center[1] * ny), 50)
        random_field_interpolated[i] = random_field[index_x, index_y]
    return random_field_interpolated



if __name__ == "__main__":
    points, tetras, mesh = read_gmsh3d()
    # triangle_areas   = get_triangle_areas_on_right_boundary(mesh)
    random_field = get_random_field(points, tetras)