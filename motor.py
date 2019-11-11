import numpy as np
from numpy.linalg import inv
from matplotlib.pyplot import *
from scipy.signal import lfilter
from control import StateSpace


#variaveis encontradas por meio dos experimentos
b = 6.465e-4
J = 2.2e-3
K=3.12e-2
L=3.72e-2
R=5.9084

dt = 0.01#tempo de discretizacao para o motor real

#modelo do sistema
A = np.array([[-R/L,-K/L],[K/J,-b/J]])
B = np.array([[1/L , 0],[0 ,-1/J]])
C = np.array([0,1])
D = np.array([0,0])

sys = StateSpace(A,B,C,D)

#discretizando o modelo
dsys = sys.sample(dt)

#Setando os dados que serao usados
Samples = 1600
tempo = np.arange(0,(Samples+1)*dt,dt)


P = np.array([[0.1389,0],[0,0.0389]])


#Sendo Q a covariancia do ruido do preocesso. O que representa a incerteza do modelo
Q = np.array([[0.009, 0],[0,0.08]])


#Sendo R ruido na medicao


R = [0.1]


#Imagina-se os estados iniciais como 0

Xk_prev = np.array([[0],[0]])


#inicializando o estado atual

Xk = []

#Entradas do Sistema
#u = np.array([[16, 0],[0, 0.021]])
u = np.array([[16 ],[0.0021]])

#Simulando o Sistema
Xreal = []
Xreal.append(dsys.A @ Xk_prev + dsys.B@u)
y = []
for i in range(1,len(tempo)):
    Xreal.append(dsys.A@Xreal[i-1]+dsys.B@u)

#####IMPLEMENTANDO O FILTRO DE KALMAN########################


#inicializando variaveis que guardarao os estados,  e as medicoes
Xk_buffer = Xk_prev
Z_buffer = []
Z_buffer.append(Xk_prev)

#matriz com o os valores medidos
H = np.array([[1, 0],[0,1]])
F = dsys.A
for i in range(0,Samples):
    #Z = ValorMedido + Ruido
    Z = Xreal[i]+5*np.random.randn()
    Z_buffer.append(Z)
    
    #Predict
    x = Xreal[i]
    P1 = F @ P @ F.transpose() + Q
    
    
    #Update
    y = Z-H @ x
    S = H @ P1 @ H.transpose() + R
    K = P1 @ H.transpose() @ inv(S)
    
    P = P1 - K @ H @ P1
    
    Xk = x + K @ y
    Xk_buffer = np.append(Xk_buffer,Xk,axis=1)
    
    Xk_prev = Xk

    
nXreal = []
nZbuffer = []
Xk_buffer = np.append(Xk_buffer,Xk,axis=1)
Z_buffer.append(Z)

for i in range(0,len(tempo)):
    nXreal.append(float(Xreal[i][1]))
    nZbuffer.append(float(Z_buffer[i][1]))

Xk_buffer = np.matrix(Xk_buffer[1])
Xk_buffer = Xk_buffer.tolist()[0]
plot(tempo,nZbuffer,tempo,Xk_buffer,tempo,nXreal)
title('Comparação entre os valores da velocidade Angular Medida, obtido pelo Filtro de Kalman e Modelada')
ylabel('Velocidade (rad/s)')
xlabel('Tempo (s)')
legend(['Velocidade Medida','Velocidade Kalman','Velocidade Modelada'])
