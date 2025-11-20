# testing/BSpline_test.py
import numpy as np
import matplotlib.pyplot as plt
from src.BSpline import BSplineBasis


class BSplineTester:
    """Test class for BSplineBasis functionality."""
    
    def __init__(self, t0=0, t1=25, n_basis=8, degree=4):
        self.spline_basis = BSplineBasis(t0=t0, t1=t1, n_basis=n_basis, degree=degree)
        self.test_coef_ones = np.ones((1, n_basis))
        self.test_coef_rand = np.array([[1.4, 0.2, 0.3, 6.2, 4.2, 2.4, 3.2, 0]])
    
    def run_tests(self, n_points=25):
        """Run all tests and return results."""
        print("Knot vector:", self.spline_basis.knots)
        
        # Evaluate basis functions
        ts, B = self.spline_basis(n_points=n_points)
        
        print(f"Basis matrix shape: {B.shape}")
        print(f"Coefficient vector shape: {self.test_coef_rand.shape}")
        
        # Compute function values
        y_ones = self.test_coef_ones @ B
        y_rand = self.test_coef_rand @ B
        
        print('y_rand:', y_rand)
        print('y_ones shape:', y_ones.shape)
        
        return ts, B, y_ones, y_rand
    
    def plot_results(self, y_ones, y_rand):
        """Plot the test results."""
        fig, axs = plt.subplots(2, 1, figsize=(8, 8), 
                               gridspec_kw={'height_ratios': [3, 1]})
        
        self._plot_functions(axs[0], y_ones, y_rand)
        self._plot_coefficients(axs[1])
        
        fig.tight_layout()
        plt.show()
    
    def _plot_functions(self, ax, y_ones, y_rand):
        """Plot the resulting functions."""
        ax.set_title('Resulting Function')
        
        # Plot lines with markers
        line1, = ax.plot(y_ones[0], label='coef: ones', linewidth=2, 
                        marker='o', markersize=7, markeredgewidth=1.5, 
                        markerfacecolor='white')
        line2, = ax.plot(y_rand[0], label='coef: rand', linewidth=2,
                        marker='o', markersize=7, markeredgewidth=1.5,
                        markerfacecolor='white')
        
        # Add translucent highlights
        x = np.arange(len(y_ones[0]))
        ax.scatter(x, y_ones[0], s=140, color=line1.get_color(), 
                  alpha=0.18, edgecolors='none')
        ax.scatter(x, y_rand[0], s=140, color=line2.get_color(), 
                  alpha=0.18, edgecolors='none')
        
        ax.set_ylabel('avg_Temp')
        ax.set_xlabel('Year')
        ax.legend()
    
    def _plot_coefficients(self, ax):
        """Plot the coefficient values."""
        ax.set_title('Corresponding Coefficients')
        
        x_coef = np.arange(len(self.test_coef_ones[0]))
        ax.scatter(x_coef, self.test_coef_ones[0], s=40, 
                  color='blue', edgecolors='none', label='coef: ones')
        ax.scatter(x_coef, self.test_coef_rand[0], s=40, 
                  color='orange', edgecolors='none', label='coef: rand')
        
        ax.set_xlabel('k')
        ax.set_ylabel('b[k]')
        ax.legend()


def main():
    """Main test function."""
    tester = BSplineTester()
    ts, B, y_ones, y_rand = tester.run_tests(n_points=25)
    tester.plot_results(y_ones, y_rand)


if __name__ == "__main__":
    main()