import numpy as np

class FEMBasis2D:
    """
    A wrapper for the FEM basis generator.
    """
    def __init__(self, nodes, triangles):
        """
        nodes: array of shape (N_nodes, 2) with x,y coordinates
        triangles: array of shape (N_triangles, 3) with node indices
        """
        self.nodes = np.array(nodes)
        self.triangles = np.array(triangles)

    def _barycentric_coords(self, point, tri_nodes):
        """
        Compute barycentric coordinates for a point inside one triangle.
        point: (x, y) tuple
        tri_nodes: (3,2) array of node coordinates

        Returns: (l1, l2, l3) barycentric coordinates
        """
        x, y = point
        x1, y1 = tri_nodes[0]
        x2, y2 = tri_nodes[1]
        x3, y3 = tri_nodes[2]

        # Compute the area (2x signed area of triangle)
        detT = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)

        if np.abs(detT) < 1e-12:
            return None  # Degenerate triangle

        l1 = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / detT
        l2 = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / detT
        l3 = 1 - l1 - l2

        return np.array([l1, l2, l3])

    def evaluate_basis(self, points):
        """
        Evaluate FEM P1 basis for a list of points.
        points: (N_points, 2) array
        Returns: Φ matrix of shape (N_points, N_nodes)
        """
        points = np.array(points)
        phi = np.zeros((len(points), len(self.nodes)))

        for p_idx, point in enumerate(points):
            # Find which triangle the point is in
            found = False
            for tri_idx, tri in enumerate(self.triangles):
                tri_nodes = self.nodes[tri]  # Triangle vertices

                # Compute barycentric coordinates
                l = self._barycentric_coords(point, tri_nodes)

                # Check if inside triangle
                if l is not None and np.all(l >= -1e-12) and np.all(l <= 1+1e-12):
                    phi[p_idx, tri] = l  # Assign basis values
                    found = True
                    break

            if not found:
                print(f"Warning: Point {point} is outside the mesh.")

        return phi
