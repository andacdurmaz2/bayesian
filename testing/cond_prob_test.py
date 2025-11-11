import numpy as np
import matplotlib.pyplot as plt
from src.cond_prob import sigma_e_given_rest
from src.BSpline import BSplineBasis



t0=0         #begin odf time period
t1=24        #End of time period
sigma_e=5    #prior 
d_e=10        #Hyperparameter
c_e=5        #Hyperparameter 
n=6          #Amount of samp lon measurement (lon €[45,50]=>6)
m=n*(t1+1-t0)       # number of all measuremtns (N*years=6*25=150)
spline_basis = BSplineBasis(t0=t0, t1=t1, n_basis=8, degree=3)
_, X = spline_basis(n_points=25)
print('X type: ',type(X))
b_mat=np.ones((n,1,8))
Y=np.ones((n,25,1))
Z,c,d=sigma_e_given_rest(c_e=c_e,m=m,d_e=d_e,Y=Y,X=X,b_mat=b_mat)
print(c,d)
samples = Z.rvs(size=5000)
print(f"Generated {len(samples)} samples")
print(f"Sample statistics - Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")

plt.hist(samples, bins=50, density=True, alpha=0.7)
x_range = np.linspace(0.01, 5, 1000)
plt.plot(x_range, Z.pdf(x_range), 'r-', linewidth=2)
plt.title('Inverse Gamma Samples vs PDF')
plt.show()