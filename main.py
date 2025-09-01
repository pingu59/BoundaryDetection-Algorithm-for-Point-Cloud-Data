import plane_extractor
import BPD
import numpy as np
import remove_wall
import open3d as o3d

def display_outlier(points, ind):
    inlier = points.select_by_index(ind)
    outlier = points.select_by_index(ind, invert=True)
    inlier.paint_uniform_color([0.7, 0.7, 0.7])
    outlier.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([inlier, outlier])

pc = o3d.io.read_point_cloud("clouds/sphere.ply")
o3d.visualization.draw_geometries([pc])

points = np.array(pc.points)
boundary = BPD.cal_boundary(points, save_filename="ceil_boundary.txt")

