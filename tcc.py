import numpy as np
from numpy.linalg import inv
from matplotlib.pyplot import *

#Setando os dados que serao usados
Samples = 99
dt = 0.1
tempo = np.arange(0,(Samples+1)*dt,dt)
VelReal = 10 #m/s


#Equacao do movimento
Xinicial = 0 #m
Xreal = Xinicial + VelReal*tempo


#Imagina-se que o trem tem seu estado inicial em 0 m e como velocidade 10 m/s

Xk_prev = np.array([[0],[VelReal]])


#inicializando o estado atual do trem

Xk = []

#Os estados do sistema sao Velocidade e Aceleracao, portanto x = [Posicao Velocidade]'
#A equacao de movimento seria Xk = A*Xk_prev + Ruido
#Xk(n) = Xk(n-1) + Vk(n-1)*dt
#sendo V estimado e nao medido
# F representa a dinamica do sistema, ou seja a propria equacao do movimento
F = np.array([[1,dt],[0,1]])

#Sendo P a variancia entre os estados, onde sera pesado a relacao entre a medida
#e a estimativa

sigma_model = 15

P = np.array([[sigma_model**2,0],[0,sigma_model**2]])


#Sendo Q a covariancia do ruido do preocesso. O que representa a incerteza do modelo
#Para esse problema sera assumido um modelo perfeito, sem aceleracao
#Qualquer aceleracao medida sera considerada ruido.

Q = np.array([[0, 0],[0,0]])


#Sendo R ruido na medicao

sigma_meas = 15

R = sigma_meas**2

#####IMPLEMENTANDO O FILTRO DE KALMAN########################


#inicializando variaveis que guardarao os estados,  e as medicoes
Xk_buffer = Xk_prev
Z_buffer = [0]

#matriz com o os valores medidos na primeira amos tra M(1) = 1
#como nao se me de velocidade temos que M(2) = 0
M = np.array([[1, 0]]);

for i in range(0,Samples):
    
    #Z = ValorMedido + Ruido
    Z = Xreal[i]+sigma_meas*np.random.rand()
    Z_buffer.append(Z)
    
    P1 = F @ P @ F.transpose() + Q
    S = M @ P1 @ M.transpose() + R
    
    K = P1 @ M.transpose() @ inv(S)
    
    P = P1 - K @ M @ P1
    
    Xk = F @ Xk_prev + K @ (Z-M @ F @ Xk_prev)
    Xk_buffer = np.append(Xk_buffer,Xk,axis=1)
    
    Xk_prev = Xk

    
plot(tempo,Xreal,tempo,Z_buffer,tempo,Xk_buffer[0])
title('Comparação entre os valores da posição Real, Medido e obtido pelo Filtro de Kalman')
ylabel('Posição (m)')
xlabel('Tempo (s)')
legend(['Posição Real','Posição Medida','Posição Kalman'])

velZ = []
Zprev = 0

for position in Z_buffer:
    valor = (position-Zprev)/dt
    Zprev = position
    velZ.append(valor)
    
    
Vreal = VelReal*np.ones(len(tempo))

figure()
plot(tempo,Vreal,tempo,velZ,tempo,Xk_buffer[1])
title('Comparação entre os valores da Velocidade Real, Medido e obtido pelo Filtro de Kalman')
ylabel('Velocidade (m/s)')
xlabel('Tempo (s)')
legend(['Velocidade Real','Velocidade Medida','Velocidade Kalman'])
        