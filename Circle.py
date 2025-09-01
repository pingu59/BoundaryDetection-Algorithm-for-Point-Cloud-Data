import numpy as np
import math

class Circle:
    def __init__(self, p1, p2, p3):
        """
        Initialize a circle from three points (automatically detects 2D or 3D)
        p1, p2, p3: numpy arrays representing points (2D or 3D)
        """
        # Convert points to numpy arrays if they aren't already
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        
        # Determine if we're working in 2D or 3D
        dim = max(p1.shape[0], p2.shape[0], p3.shape[0])
        
        if dim == 2:
            # Handle 2D case
            self._init_2d(p1, p2, p3)
        else:
            # Handle 3D case
            self._init_3d(p1, p2, p3)
    
    def _init_2d(self, p1, p2, p3):
        """Initialize circle from three 2D points"""
        # Ensure all points are 2D
        p1 = p1[:2]
        p2 = p2[:2]
        p3 = p3[:2]
        
        v1 = np.dot(p2, p2)
        v2 = (np.dot(p1, p1) - v1) / 2
        v3 = (v1 - (p3[0]**2) - (p3[1]**2)) / 2
        det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

        if abs(det) < 1.0e-10:
            self.radius = None
            self.center = None
            self.normal = None
            return

        # Center of circle
        cx = round((v2 * (p2[1] - p3[1]) - v3 * (p1[1] - p2[1])) / det, 5)
        cy = round(((p1[0] - p2[0]) * v3 - (p2[0] - p3[0]) * v2) / det, 5)

        self.center = np.array([cx, cy])
        self.radius = round(np.linalg.norm(p1 - self.center), 5)
        self.normal = None  # No normal in 2D case
    
    def _init_3d(self, p1, p2, p3):
        """Initialize circle from three 3D points"""
        # Ensure all points are 3D
        p1 = p1 if len(p1) == 3 else np.append(p1, 0)
        p2 = p2 if len(p2) == 3 else np.append(p2, 0)
        p3 = p3 if len(p3) == 3 else np.append(p3, 0)
        
        # Check if points are collinear
        v1 = p2 - p1
        v2 = p3 - p1
        cross_product = np.cross(v1, v2)
        
        if np.linalg.norm(cross_product) < 1.0e-10:
            self.radius = None
            self.center = None
            self.normal = None
            return
        
        # Find the plane containing the three points
        self.normal = cross_product / np.linalg.norm(cross_product)
        
        # Project points onto the plane for 2D circle calculation
        # Create a coordinate system in the plane
        u = v1 / np.linalg.norm(v1)
        w = np.cross(self.normal, u)
        w = w / np.linalg.norm(w)
        
        # Convert 3D points to 2D coordinates in the plane
        p1_2d = np.array([0, 0])
        p2_2d = np.array([np.dot(p2 - p1, u), 0])
        p3_2d = np.array([np.dot(p3 - p1, u), np.dot(p3 - p1, w)])
        
        # Calculate circle in 2D
        v1_2d = np.dot(p2_2d, p2_2d)
        v2_2d = (np.dot(p1_2d, p1_2d) - v1_2d) / 2
        v3_2d = (v1_2d - (p3_2d[0]**2) - (p3_2d[1]**2)) / 2
        
        det = (p1_2d[0] - p2_2d[0]) * (p2_2d[1] - p3_2d[1]) - (p2_2d[0] - p3_2d[0]) * (p1_2d[1] - p2_2d[1])
        
        if abs(det) < 1.0e-10:
            self.radius = None
            self.center = None
            return
        
        # Center of circle in 2D coordinates
        cx_2d = (v2_2d * (p2_2d[1] - p3_2d[1]) - v3_2d * (p1_2d[1] - p2_2d[1])) / det
        cy_2d = ((p1_2d[0] - p2_2d[0]) * v3_2d - (p2_2d[0] - p3_2d[0]) * v2_2d) / det
        
        # Convert 2D center back to 3D
        center_2d = np.array([cx_2d, cy_2d])
        self.center = p1 + cx_2d * u + cy_2d * w
        self.radius = np.linalg.norm(p1_2d - center_2d)
        
        # Round for consistency
        self.center = np.round(self.center, 5)
        self.radius = round(self.radius, 5)

    def print(self):
        """Print circle properties"""
        if self.radius is None:
            print("Invalid circle (points are collinear)")
        else:
            if self.normal is None:
                print(f'R: {self.radius} C: {self.center}')
            else:
                print(f'R: {self.radius} C: {self.center} Normal: {self.normal}')

    def to_array(self, num_points=36):
        """
        Generate points on the circle
        num_points: number of points to generate around the circle
        """
        if self.radius is None or self.center is None:
            return None
        
        if self.normal is None:
            # 2D case
            angles = np.linspace(0, 2 * math.pi, num_points, endpoint=False)
            points = np.array([
                [self.center[0] + self.radius * math.cos(angle),
                 self.center[1] + self.radius * math.sin(angle)]
                for angle in angles
            ])
            return points
        else:
            # 3D case
            # Create two orthogonal vectors in the plane
            if abs(self.normal[0]) > abs(self.normal[1]):
                u = np.array([self.normal[2], 0, -self.normal[0]])
            else:
                u = np.array([0, self.normal[2], -self.normal[1]])
            u = u / np.linalg.norm(u)
            v = np.cross(self.normal, u)
            v = v / np.linalg.norm(v)
            
            # Generate points around the circle
            points = []
            for i in range(num_points):
                angle = 2 * math.pi * i / num_points
                point = self.center + self.radius * (math.cos(angle) * u + math.sin(angle) * v)
                points.append(point)
            
            return np.array(points)