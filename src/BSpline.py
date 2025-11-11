import numpy as np
from scipy.interpolate import BSpline
from typing import List, Tuple, Optional

class BSplineBasis:
    """
    A wrapper for the B-spline basis generator.
    """
    
    def __init__(self, t0: float, t1: float, n_basis: int = 8, degree: int = 3):
        """
        Initialize B-spline basis.
        
        Parameters:
        -----------
        t0 : float
            Start of time range
        t1 : float
            End of time range  
        n_basis : int
            Number of basis functions (default: 8)
        degree : int
            Degree of spline (default: 3 for cubic)
        """
        self.t0 = t0
        self.t1 = t1
        self.n_basis = n_basis
        self.degree = degree
        
        self._setup_knots()
        self._setup_basis_functions()
    
    def _setup_knots(self) -> None:
        """Setup the knot vector with uniform internal knots."""
        n_internal = self.n_basis - self.degree - 1
        
        if n_internal < 0:
            raise ValueError(f"Number of basis functions ({self.n_basis}) must be at least degree+2 ({self.degree + 2})")
        
        # Internal knots uniformly spaced
        internal_knots = np.linspace(self.t0, self.t1, n_internal + 2)[1:-1]
        
        # Full knot vector with multiplicity degree+1 at boundaries
        self.knots = np.concatenate((
            np.full(self.degree + 1, self.t0),
            internal_knots,
            np.full(self.degree + 1, self.t1)
        ))
    
    def _setup_basis_functions(self) -> None:
        """Create the basis functions as BSpline objects."""
        # Coefficient matrix: identity -> each row is one basis function
        C = np.eye(self.n_basis)
        
        # Generate basis functions
        self.basis_functions = [BSpline(self.knots, C[i], self.degree) 
                               for i in range(self.n_basis)]
    
    def evaluate(self, t_values: Optional[np.ndarray] = None, 
                 n_points: int = 25) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate all basis functions at given points or on a uniform grid.
        
        Parameters:
        -----------
        t_values : np.ndarray, optional
            Specific points to evaluate at. If None, creates uniform grid.
        n_points : int
            Number of points for uniform grid (if t_values not provided)
            
        Returns:
        --------
        t_eval : np.ndarray
            Evaluation points
        B : np.ndarray 
            Basis matrix of shape (n_basis, len(t_eval))
        """
        if t_values is None:
            t_eval = np.linspace(self.t0, self.t1, n_points)
        else:
            t_eval = np.asarray(t_values)
        
        # Evaluate all basis functions and stack
        B = np.vstack([bf(t_eval) for bf in self.basis_functions])
        
        return t_eval, B
    
    def get_basis_function(self, i: int) -> BSpline:
        """Get the i-th basis function."""
        if i < 0 or i >= self.n_basis:
            raise IndexError(f"Basis function index {i} out of range [0, {self.n_basis-1}]")
        return self.basis_functions[i]
    
    def __call__(self, t_values: Optional[np.ndarray] = None, 
                 n_points: int = 25) -> Tuple[np.ndarray, np.ndarray]:
        """Alias for evaluate method."""
        return self.evaluate(t_values, n_points)
    
    def __len__(self) -> int:
        return self.n_basis
    
    def __repr__(self) -> str:
        return f"BSplineBasis(t0={self.t0}, t1={self.t1}, n_basis={self.n_basis}, degree={self.degree})"

