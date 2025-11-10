# testing/BSpline_test.py
from Code.BSpline import BSplineBasis
import numpy as np
import matplotlib.pyplot as plt

# Test the class
if __name__ == "__main__":
    spline_basis = BSplineBasis(t0=0, t1=25, n_basis=8, degree=3)
    print("Knot vector:", spline_basis.knots)
    
    # Test evaluation
    ts, B = spline_basis(n_points=25)
    test_coef_ones=np.ones((1,8))
    test_coef_rand=np.array([[1.4,0.2,0.3,6.2,4.2,2.4,3.2,0]])
    print(f"Basis matrix shape: {B.shape}")
    print(f"Coefficient vector shape: {test_coef_rand.shape}")
    y_ones=test_coef_ones @B
    y_rand=test_coef_rand@B
    print('y:',y_rand)
    print('Output vector:',y_ones.shape)


    fig, axs = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1]})
    ax = axs[0]
    ax.set_title('Resulting function')
    # plot lines with bigger markers
    line1, = ax.plot(y_ones[0], label='coef: ones', linewidth=2, marker='o', markersize=7, markeredgewidth=1.5, markerfacecolor='white')
    line2, = ax.plot(y_rand[0], label='coef: rand', linewidth=2, marker='o', markersize=7, markeredgewidth=1.5, markerfacecolor='white')
    # add a fatter translucent highlight behind each point
    x = np.arange(len(y_ones[0]))
    ax.scatter(x, y_ones[0], s=140, color=line1.get_color(), alpha=0.18, edgecolors='none')
    ax.scatter(x, y_rand[0], s=140, color=line2.get_color(), alpha=0.18, edgecolors='none')
    ax.set_ylabel('avg_Temp')
    ax.set_xlabel('Year')
    ax.legend()

    ax2=axs[1]
    x_2=np.arange(0,8)
    print('x_2:',x_2.shape,'ones',test_coef_ones[0].shape)
    ax2.set_title('Corresponding Coefficients')
    ax2.scatter(x_2, test_coef_ones[0], s=40, color=line1.get_color(),edgecolors='none',label='coef: ones')
    ax2.scatter(x_2, test_coef_rand[0], s=40, color=line2.get_color(),edgecolors='none',label='coef: rand')
    
    

    ax2.set_xlabel('k')
    ax2.set_ylabel('b[k]')
    
    
    ax2.legend()

    fig.tight_layout()
    plt.show()
    