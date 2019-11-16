from kf_book.book_plots import figsize
import kf_book.book_plots as book_plots
import matplotlib.pyplot as plt
import random
from filterpy.discrete_bayes import normalize
from filterpy.discrete_bayes import update
import numpy as np
from filterpy.discrete_bayes import predict


x = np.arange(len([0]*13))
width = 0.3

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def lh_hallway(hall, z, z_prob):
       
    try:
        scale = z_prob / (1. - z_prob)
    except ZeroDivisionError:
        scale = 1e8

    likelihood = np.ones(len(hall))
    likelihood[hall==z] *= scale

    return likelihood


class Train(object):

    def __init__(self, track_len, kernel=[1.], sensor_accuracy=.9):
        self.track_len = track_len
        self.pos = 0
        self.kernel = kernel
        self.sensor_accuracy = sensor_accuracy

    def move(self, distance=1):
        """ move in the specified direction
        with some small chance of error"""

        self.pos += distance
        # insert random movement error according to kernel
        r = random.random()
        s = 0
        offset = -(len(self.kernel) - 1) / 2
        for k in self.kernel:
            s += k
            if r <= s:
                break
            offset += 1
        self.pos = int((self.pos + offset) % self.track_len)
        return self.pos

    def sense(self):
        pos = self.pos
         # insert random sensor error
        if random.random() > self.sensor_accuracy:
            if random.random() > 0.5:
                pos += 1
            else:
                pos -= 1
        return pos

def train_filter(iterations, kernel, sensor_accuracy, 
             move_distance, do_print=True):
    track = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12])
    prior = np.array([1.] + [0]*12)
    posterior = prior[:]
    normalize(prior)
    
    robot = Train(len(track), kernel, sensor_accuracy)
    for i in range(iterations):
        # move the robot and
        robot.move(distance=move_distance)
        
        # peform prediction
        prior = predict(posterior, move_distance, kernel)#preve o sistema

#        print(prior)
        #  and update the filter
        m = robot.sense()
        likelihood = lh_hallway(track, m, sensor_accuracy)
        posterior = update(likelihood, prior)
        index = np.argmax(posterior)
        
        if do_print:
            print('''time {}: pos {}, sensed {}, '''
                  '''at position {}'''.format(
                    i, robot.pos, m, track[robot.pos]))
            
            print('''        estimated position is {}'''
                  ''' with confidence {:.4f}%:'''.format(
                  index, posterior[index]*100))            

#    book_plots.bar_plot(prior, ylim=(0, 1))
    if iterations > 0:
        plt.bar(x-width/2,posterior,width,label='Filtro')
#        plt.bar(x,normalize(likelihood),width,label='Medição')
        plt.bar(x+width/2,prior,width,label='Previsao')
    else:
        m = 0    
        plt.bar(x,prior,width,label='Previsão')
    
    if do_print:
        print()
        print('final position is', robot.pos)
        index = np.argmax(posterior)
        print('''Estimated position is {} with '''
              '''confidence {:.4f}%:'''.format(
                index, posterior[index]*100))
    
    return m

index = [1,2]
index.extend([1,2,3,4]*3)   
#index = [1,2,3,4]*3
with figsize(y=5.5):
    j=1
    flag = False
   
    for indice,i in enumerate(index):
        random.seed(3)
        if i == 1:
            plt.figure()
        elif indice == 14:
            break
        if indice == 0 or indice == 1:
            a = plt.subplot(1,2,i)
        else:
            a = plt.subplot(2,2,i)
        m = train_filter(indice, kernel=[.05,.1, .7, .1 ,.05], 
                     sensor_accuracy=.7,
                     move_distance=1, do_print=False)
       
        plt.title ('Iteração {}, o sensor mediu: {} metros'.format(indice,m))
        plt.legend()
        plt.xlabel('Posição')
        plt.ylabel('Probabilidade')
        plt.tight_layout()