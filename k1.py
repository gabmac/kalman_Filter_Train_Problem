from collections import namedtuple
from numpy.random import randn
from kf_book import book_plots
import numpy as np
import matplotlib.pyplot as plt



gaussian = namedtuple('Gaussian', ['mean', 'var'])

def distancia( std,N):
    dist = []
    for i in range(0,N):
        dist.append(i + (randn() * std))
        
    return dist 

def predict(posterior, movement):
    x, P = posterior # mean and variance of posterior
    dx, Q = movement # mean and variance of movement
    x = x + dx
    P = P + Q
    return gaussian(x, P)

def update(prior, measurement):
    x, P = prior        # mean and variance of prior
    z, R = measurement  # mean and variance of measurement
    
    y = z - x        # residual
    K = P / (P + R)  # Kalman gain

    x = x + K*y      # posterior
    P = (1 - K) * P  # posterior variance
    return gaussian(x, P)

distance_std = 1
process_var = 0.7**2

x = gaussian(0., 1000.) # initial state
process_model = gaussian(1., process_var)

N = 12
zs = distancia(distance_std,N)
ps = []
estimates = []
priors = np.zeros((N,2))
for i,z in enumerate(zs):
    prior = predict(x, process_model)
    priors[i] = prior
    x = update(prior, gaussian(z, distance_std**2))

    # save for latter plotting
    estimates.append(x.mean)
    ps.append(x.var)

# plot the filter output and the variance
book_plots.plot_measurements(zs)
book_plots.plot_filter(estimates, var=np.array(ps))
book_plots.plot_predictions(priors[:,0])
book_plots.show_legend()
book_plots.set_labels(x='Tempo (s)', y='Posições')
plt.show()

plt.figure()   
plt.plot(ps)
plt.title('Variância')
print('Variance converges to {:.3f}'.format(ps[-1]))
