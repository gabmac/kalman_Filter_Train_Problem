import numpy as np
from numpy.linalg import inv
from matplotlib.pyplot import *
from scipy.signal import lfilter

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


P = np.array([[10**2,0],[0,33**2]])


#Sendo Q a covariancia do ruido do preocesso. O que representa a incerteza do modelo
#Para esse problema sera assumido um modelo perfeito, sem aceleracao
#Qualquer aceleracao medida sera considerada ruido.

Q = np.array([[0.588, 1.175],[1.175,2.35]])


#Sendo R ruido na medicao

sigma_meas = 10

R = np.array(sigma_meas**2)

#####IMPLEMENTANDO O FILTRO DE KALMAN########################


#inicializando variaveis que guardarao os estados,  e as medicoes
Xk_buffer = Xk_prev
Z_buffer = [0]

#matriz com o os valores medidos H(1) = 1
#como nao se me de velocidade temos que H(2) = 0
H = np.array([[1, 0]]);
teste = []
for i in range(0,Samples):
    
    #Z = ValorMedido + Ruido
    Z = Xreal[i]+sigma_meas*np.random.randn()
    Z_buffer.append(Z)
    
    #Predict
    x = F @ Xk_prev
    P1 = F @ P @ F.transpose() + Q
    
    
    #Update
    y = Z-H @ x
    teste.append(y)
    S = H @ P1 @ H.transpose() + R
    K = P1 @ H.transpose() @ inv(S)
    
    P = P1 - K @ H @ P1
    
    Xk = x + K @ y
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

windowSize = 5

velMedia = lfilter(np.ones(windowSize) / windowSize, 1, velZ)

figure()
plot(tempo,Vreal,tempo,velZ,tempo,Xk_buffer[1],tempo,velMedia)
title('Comparação entre os valores da Velocidade Real, Medido e obtido pelo Filtro de Kalman')
ylabel('Velocidade (m/s)')
xlabel('Tempo (s)')
legend(['Velocidade Real','Velocidade Medida','Velocidade Kalman','Velocidade Média do Sensor'])
        
