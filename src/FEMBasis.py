import numpy as np
from scipy.spatial import Delaunay

class FEMBasis2D:
    """
    FEM P1 basis (meaning linear hat functions, we can still change for more smoothness later) for 2D triangular mesh, with support for generating
    a reduced number of basis nodes (e.g., uniform grid).
    """

    def __init__(self, nodes, triangles):
        self.nodes = np.array(nodes)        # (N_nodes, 2 -> (x,y))
        self.triangles = np.array(triangles)  # (N_triangles, 3 -> (which 3 vertice))
        self.tri_engine = Delaunay(self.nodes)  # for fast lookup tri_engine: a Delaunay object built on nodes to quickly find: Which triangle contains a given point (find_simplex).

    @classmethod



    def from_domain(cls, domain, K, exact=False):
        """
        makes a unifrom grid with the K amount of nodes we want to have

        K = amount of basis we want to have, corresponding to the amount of nodes
        Generate K (approximately) basis nodes uniformly over the domain.
        domain = ((xmin, ymin), (xmax, ymax))
        """
        (xmin, ymin), (xmax, ymax) = domain

        # Choose grid resolution (≈ square)
        nx = int(np.floor(np.sqrt(K)))
        ny = int(np.ceil(K / nx))


        #creates grid
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(xs, ys)
        nodes = np.vstack([X.ravel(), Y.ravel()]).T # node and coordinate


        if exact and len(nodes) > K:
            nodes = nodes[:K]  # trim to exactly K basis nodes

        #Uses Delaunay triangulation to create triangles from the nodes.
        triangles = Delaunay(nodes).simplices
        return cls(nodes, triangles)

    @staticmethod
    def _barycentric_coords(point, tri_nodes):
        x, y = point
        x1, y1 = tri_nodes[0]
        x2, y2 = tri_nodes[1]
        x3, y3 = tri_nodes[2]

        detT = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
        if abs(detT) < 1e-12:
            return None

        l1 = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / detT
        l2 = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / detT
        l3 = 1 - l1 - l2
        return np.array([l1, l2, l3])

    def evaluate_basis(self, points):
        """
        Returns Φ (n_points × n_basis)
        """
        points = np.array(points)
        phi = np.zeros((len(points), len(self.nodes)))

        simplex_idx = self.tri_engine.find_simplex(points)

        for i, point in enumerate(points):
            s = simplex_idx[i]
            if s == -1:
                continue  # outside domain

            tri = self.triangles[s]
            tri_nodes = self.nodes[tri]
            bary = self._barycentric_coords(point, tri_nodes)

            if bary is not None and np.all(bary >= -1e-12):
                phi[i, tri] = bary

        return phi

    def __len__(self):
        return len(self.nodes)
