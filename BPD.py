import numpy as np
import Circle as C
import open3d as o3d
import polar
from sklearn.neighbors import NearestNeighbors
from itertools import combinations

def cal_boundary(points, k=30, save_filename=None) :
    neighbors = NearestNeighbors(n_neighbors=k).fit(points)
    distances, indices = neighbors.kneighbors(points)

    max_b = 0
    min_b = 1000000000

    boundary = []
    for i in range(points.shape[0]) :
        is_boundary = False
        p_neighbor, p_distance = indices[i], np.round(distances[i], 5)
        neighbor_points = points[p_neighbor[1:]]

        p_mean = np.mean(p_distance)
        p_std = np.std(p_distance)
        local_resol = round(p_mean + 2 * p_std, 5)

        if local_resol > max_b :
            max_b = local_resol
        if min_b > local_resol :
            min_b = local_resol

        pairs = list(combinations(p_neighbor[1:], 2))
        print(i)
        for j in range(len(pairs)) :
            count = 0
            p1 = points[i]
            p2 = points[pairs[j][0]]
            p3 = points[pairs[j][1]]
            c = C.Circle(p1, p2, p3)
            if c.radius == None :
                continue

            if c.radius >= local_resol :
                cn_distance = np.linalg.norm((neighbor_points - c.center), axis=1)
                cn_distance = np.round(cn_distance, 5)

                for k in range(len(cn_distance)) :
                    if cn_distance[k] <= c.radius :
                        count += 1

                        if count > 3 :
                            break

            if count == 3 :
                boundary.append(points[i])
                is_boundary = True
                break

        if not is_boundary :
            pol = polar.Polar(np.array(points[i]), neighbor_points, normalize=True)



    print(f'len : {len(boundary)}')
    print(f'{min_b}')
    print(f'{max_b}')
    # Convert boundary to numpy array
    boundary = np.array(boundary)
    
    # Create Open3D point cloud for visualization
    pc = o3d.geometry.PointCloud()
    
    # Check if points are 2D or 3D and handle accordingly
    if boundary.shape[1] == 2:
        # Convert 2D points to 3D by adding zero z-coordinate
        boundary_3d = np.pad(boundary, ((0, 0), (0, 1)), 'constant', constant_values=0)
        pc.points = o3d.utility.Vector3dVector(boundary_3d)
    else:
        # Already 3D points
        pc.points = o3d.utility.Vector3dVector(boundary)
    
    # Color the boundary points (optional - makes them stand out)
    pc.paint_uniform_color([1, 0, 0])  # Red color for boundary points
    
    # Visualize the boundary points
    o3d.visualization.draw_geometries([pc])
    
    # Save to file if requested
    if save_filename is not None:
        np.savetxt(save_filename, boundary)

    return boundary
