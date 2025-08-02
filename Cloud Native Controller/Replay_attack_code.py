
from Controller import LQ
from BBinterface import BBsimEulerInterface
from types import SimpleNamespace
import matplotlib.pyplot as plt
import time
import types
import numpy as np
import math
from scipy import signal
from scipy.signal import lfilter
from sigma_gamma_tilde import Sigma_gamma


#from queue import Queue
from multiprocessing import Process, Value, Array, Lock, Queue

#print("Number of cpu : ", multiprocessing.cpu_count())

#t = np.linspace(0, 10, 200, endpoint=True)
H = 0.05
SET_POINT = 0.3
#loop=3950
loop=3950
Real_pos=[]
Real_speed=[]
Real_ang=[]
Real_beamspeed=[]

cloud_pos=[]
cloud_speed=[]
cloud_ang=[]
cloud_beamspeed=[]

attacker_pos=[]
attacker_ang=[]
attacker_speed=[]
attacker_beamspeed=[]

ref=[]
GG=[]


plant = BBsimEulerInterface(h=H)

##########
#Kalman:
y_before=np.array([[0],[0]])
yhat_before=np.array([[0],[0]])
#u=0
x_hat_k_k_1=np.array([[0],[0],[0]])
x_hat_k_k=np.array([[0],[0],[0]])

K=np.array([ [0.0474, 0.0321, -0.0004], [0.0321, 0.0808, -0.0114], [-0.0004, -0.0114, 0.0035] ]) #kalman gain

P=np.array([ [0.1019e-03, 0.0735e-3, -0.0016e-03], [0.0735e-03, 0.1788e-03, -0.0251e-03], [-0.0016e-03, -0.0251e-03, 0.0072e-03]])

R=np.array([ [0.002, 0, 0], [0, 0.002, 0], [0, 0, 0.002] ])

A = np.array([[ 1.,0.05,-0.008756],[0.,1.,-0.3502],[0.,0.,1.]])

B = np.array([[-6.421e-05], [-0.003853], [0.022]])

C = np.array([[1,0.,0.], [0, 1, 0], [0., 0., 1]])

D = np.array([[0.], [0.], [0.]])

cov_e = 10
#cov_e = 13

#sigma_gamma_tilde= np.array( [[131.75113653, -30.37558696, 40.64427924], [-30.37558696, 122.50577283, 1.60866034], [ 40.64427924, 1.60866034, 57.15137524]] )
sigma_gamma_tilde=Sigma_gamma.sigma(cov_e)



T_s=0.05
sigma_w = 0.05     # Process noise
sigma_v = 1e-2     # Measurement noise
# Variances need to be scaled for the discrete time Kalman Filter  ?????
var_v_d = math.pow(sigma_v,2)/T_s
var_w_d = math.pow(sigma_w,2)*T_s
Q = var_w_d
R = np.eye(np.size(C,0))*var_v_d

Q=np.eye(np.size(A,0))*var_w_d #??????



#LQ controller coefficient
L=np.array([[31.0027, 13.9077, -20.4937]])  ### need to be checked

#import control
sigma_e=cov_e
#sigma_e=1

e=np.random.normal(0,math.sqrt(sigma_e),loop)


############

sigma_gamma=np.dot(np.dot(C,P),np.transpose(C))+R


sigma_gamma_e=np.concatenate([ np.concatenate( [sigma_gamma, np.zeros((3,1)) ], axis = 1), np.concatenate( [np.zeros((1,3)), [[sigma_e]] ], axis = 1)], axis=0)

print('sigma_gamma_e: ',sigma_gamma_e)


#aa=sigma_gamma_tilde  # need to calculate this
aa=np.eye(3)

bb=-np.dot(np.dot(C,B),np.array(sigma_e))
#print('bb:',bb)

cc=-np.dot(np.dot(sigma_e,np.transpose(B)),np.transpose(C))

dd=sigma_e

sigma_gamma_e_tilde=np.concatenate([ np.concatenate( [sigma_gamma_tilde, -np.dot(np.dot(C,B),sigma_e) ], axis = 1), np.concatenate( [-np.dot(np.dot(sigma_e,np.transpose(B)),np.transpose(C)), [[sigma_e]] ], axis = 1)], axis=0)
print(sigma_gamma_e_tilde)


#############

controller = LQ()

g=0

############




Ramp_Attack=np.linspace(0,60,1200) # Attack
e=np.random.normal(0,math.sqrt(sigma_e),loop)

for i in range(loop):

    atime1=time.time()

    sensor_input = plant.read()
        #sensor_input['set_point'] = SET_POINT
    sensor_input['set_point'] = 0.35
        #sensor_input['set_point'] = 0.35*signal.square(0.02 * np.pi  * i*H)
    #ref.append(0.35*signal.square(0.02 * np.pi  * i*H))

    n = SimpleNamespace(**sensor_input)

    #### adding noise to measurements
    V=np.random.normal(0., 0.01, (4,1)) # measurement noise (standard deveiation: 0.0316)
    n.pos=n.pos+V[0]
    n.ang=n.ang+V[1]
    n.speed=n.speed+V[2]
    n.beamspeed=n.beamspeed+V[3]

    Real_pos.append(n.pos)
    Real_speed.append(n.speed)
    Real_ang.append(n.ang)
    Real_beamspeed.append(n.beamspeed)
        
    print('i',i,'pos=', n.pos, 'ang=', n.ang, 'speed=', n.speed, 'beamspeed=', n.beamspeed)

    if i>=1100 and i<=1950:
        attacker_pos.append(n.pos)
        attacker_ang.append(n.ang)
        attacker_speed.append(n.speed)
        attacker_beamspeed.append(n.beamspeed)
    else:
        pass
        
    if i>=3100 and i<=3950:

        n.pos=attacker_pos[(i-3100)]
        n.ang=attacker_ang[(i-3100)]
        n.speed=attacker_speed[(i-3100)]
        n.beamspeed=attacker_beamspeed[(i-3100)]
    else:
        pass
        
    cloud_pos.append(n.pos)
    cloud_ang.append(n.ang)
    cloud_speed.append(n.speed)
    cloud_beamspeed.append(n.beamspeed)

        #############
    y=np.array([n.pos,n.speed,n.ang])

    print('e: ', e[i])

    #global x_hat_k_k_1
    #global x_hat_k_k
    #global y_before
    #global yhat_before

    #gamma=y_before - yhat_before

    u_before=float(n.beamspeed)/0.44

    x_hat_k_k_1 = np.dot(A,x_hat_k_k) + np.dot(B,u_before) 

    #print('np.dot(A,x_hat_k_k) : ',np.dot(A,x_hat_k_k) )
    #print('np.dot(B,u_before) : ', np.dot(B,u_before) )
    #print('x_hat_k_k_1: ', x_hat_k_k_1)

    gamma=y-np.dot(C,x_hat_k_k_1)

    #print('np.dot(C,x_hat_k_k_1) : ', np.dot(C,x_hat_k_k_1))
    #print('y: ', y)

    x_hat_k_k=x_hat_k_k_1+ np.dot(K,gamma)

    #y_hat = np.dot(C,x_hat)
    #yhat_before=y_hat

    #r=y-y_hat
    #r1=r[0,0]
    #r2=r[1,0]

    #y_before=y

    #a=float(controller.update(n))+e[i]
    #a=controller.uf_prediction(n)

    #global u

    #u=float(a[0])

    print('gamma: ', gamma)

    gamma_ek_vertical=np.array([gamma[0], gamma[1], gamma[2], [e[i-1]]])

    gamma_ek_Horizonta=np.array([gamma[0,0], gamma[1,0], gamma[2,0], e[i-1]])

    print('horizon: ',gamma_ek_Horizonta, 'vertical: ', gamma_ek_vertical)



    #sigma_gamma_e_tilde=np.array( [ [aa[0,0], aa[0,1], bb[0,0]], [ aa[1,0], aa[1,1], bb[1,0] ], [cc[0,0], cc[0,1], sigma_e ] ]  )


    Anum_tilde=np.dot(-0.5, np.dot(np.dot(gamma_ek_Horizonta, np.linalg.inv(sigma_gamma_e_tilde)), gamma_ek_vertical))

    Bnum=np.dot(-0.5, np.dot(np.dot(gamma_ek_Horizonta, np.linalg.inv(sigma_gamma_e)), gamma_ek_vertical))

    print('A:', Anum_tilde, 'B: ', Bnum)

    #global g

    #print('try::', math.sqrt(np.linalg.det(sigma_gamma_e)/np.linalg.det(sigma_gamma_e_tilde)))

    #I added abs here to make it positive for sqrt

    #Anum_tilde-Bnum
    print('a-b: ', (Anum_tilde-Bnum))
    #print('Anum_tilde-Bnum: ', max(0, (math.log(math.sqrt(abs(np.linalg.det(sigma_gamma_e)/np.linalg.det(sigma_gamma_e_tilde))))+(Anum_tilde-Bnum))) )


    print(f'added test_statistics term:', math.log(math.sqrt(abs(np.linalg.det(sigma_gamma_e)/np.linalg.det(sigma_gamma_e_tilde))))+(Anum_tilde-Bnum))
    g=max(0, (g+math.log(math.sqrt(abs(np.linalg.det(sigma_gamma_e)/np.linalg.det(sigma_gamma_e_tilde))))+(Anum_tilde-Bnum)) )
    #ag=math.log(math.sqrt(abs(np.linalg.det(sigma_gamma_e)/np.linalg.det(sigma_gamma_e_tilde))))
    #ag=ag+(Anum_tilde-Bnum) 
    #g=ag

    GG.append(g)

    ctrl = controller.update(n)+e[i]

    t2=time.time()

    #exe_time=t2-t1

    ####################

        

    if i>=3100 and i<=3950:

        ctrl = ctrl+Ramp_Attack[i-3100]

    else:
        pass



    plant.act(value=ctrl)
    plant.nextstep()



# plt.figure()
# plt.plot(GG)
# plt.title('g')
# plt.xlabel('Time Slot')
# plt.ylabel('CUSUM Test Statistics')



plt.figure()
plt.plot(Real_pos)
plt.title('real pos')

plt.figure()
plt.plot(Real_speed)
plt.title('real speed')

plt.figure()
plt.plot(Real_ang)
plt.title('real angle')

plt.figure()
plt.plot(cloud_pos)
plt.title('cloud pos')

plt.figure()
plt.plot(cloud_speed)
plt.title('cloud speed')

plt.figure()
plt.plot(cloud_ang)
plt.title('cloud angle')


plt.show()