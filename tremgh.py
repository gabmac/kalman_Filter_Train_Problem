from kf_book.gh_internal import plot_g_h_results
import kf_book.book_plots as book_plots
import matplotlib.pylab as pylab
from numpy.random import randn
import numpy as np
from matplotlib.pyplot import *

def gen_data(x0, dx, count, noise_factor):
    return [x0 + dx*i + randn()*noise_factor for i in range(count)]

def g_h_filterForTrain(data, x0, dx, g, h, dt=1):
    x_est = x0
    results = []
    pred = []
    for z in data:
        # prediction step
        x_pred = x_est + (dx*dt)
        pred.append(x_pred)
        print(x_pred)
        dx = dx

        # update step
        residual = z - x_pred
        dx = dx + h * (residual) / dt
        x_est = x_pred + g * residual
        results.append(x_est)
        
    return results,pred

amostras = 12
weights = gen_data(x0=0,dx=1,count=amostras,noise_factor=0.7)
#book_plots.plot_track([0, 30], [0, 30], label='Actual weight')
data,pred = g_h_filterForTrain(data=weights, x0=0., dx=1., g=5./10, h=5./10, dt=1)
#plot_g_h_results(weights, data)
tempo = [x/1 for x in range(0,amostras)]
plot(tempo,weights,'ko',tempo,data,tempo,pred,'--')
title('Comparação entre as Posições Encontradas após 12 Segundos de Movimento')
ylabel('Posição')
xlabel('Tempo (s)')
legend(['Posições Medidas','Posições do filtro G-H','Posições Previstas'])
