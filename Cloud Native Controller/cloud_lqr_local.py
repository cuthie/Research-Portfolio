from aiohttp import web
import asyncio
import aiohttp
from types import SimpleNamespace
from numpy import linalg as la
import json
import matplotlib.pyplot as plt

import argparse
from asyncio import Queue
import socket
import sys
import os

from Controller import LQ
from BBinterface import BBsimEulerInterface
from types import SimpleNamespace
import time
import types
import numpy as np
import math
from scipy import signal
from scipy.signal import lfilter
from sigma_gamma_tilde import Sigma_gamma

# Parameter Initialization
N = 3950
# Real_pos = []
# Real_speed = []
# Real_ang = []
# Real_beamspeed = []
#
# cloud_pos = []
# cloud_speed = []
# cloud_ang = []
# cloud_beamspeed = []

attacker_pos = []
attacker_ang = []
attacker_speed = []
attacker_beamspeed = []

# Test Statistics Initialization
g = 0

# ref = []
# GG = []

MY_PORT = 60031
#MY_PORT = 8080



# Kalman Gain
K = np.array([[0.0474, 0.0321, -0.0004], [0.0321, 0.0808, -0.0114], [-0.0004, -0.0114, 0.0035]])
P = np.array([[0.1019e-03, 0.0735e-3, -0.0016e-03], [0.0735e-03, 0.1788e-03, -0.0251e-03], [-0.0016e-03, -0.0251e-03, 0.0072e-03]])
#R = np.array([[0.002, 0, 0], [0, 0.002, 0], [0, 0, 0.002]])

# System Parameters
A = np.array([[1., 0.05, -0.008756], [0., 1., -0.3502], [0., 0., 1.]])
B = np.array([[-6.421e-05], [-0.003853], [0.022]])
C = np.array([[1, 0., 0.], [0, 1, 0], [0., 0., 1]])
D = np.array([[0.], [0.], [0.]])

# Other Parameters
T_s = 0.05
sigma_w = 0.05    # Process noise
sigma_v = 1e-2     # Measurement noise

# Variances need to be scaled for the discrete time Kalman Filter
var_v_d = math.pow(sigma_v, 2)/T_s
var_w_d = math.pow(sigma_w, 2)*T_s
#Q = var_w_d
R = np.eye(np.size(C, 0))*var_v_d
Q = np.eye(np.size(A, 0))*var_w_d

# Sigma_e Initialization
#cov_e = 10
#cov_e = 0.0001
cov_e = 1e-20
sigma_gamma_tilde = Sigma_gamma.sigma(cov_e)

#LQ controller coefficient
L = np.array([[31.0027, 13.9077, -20.4937]])

# Sigma_e Initialization
sigma_e = cov_e

# Sigma_gamma Computation
sigma_gamma = np.dot(np.dot(C, P), np.transpose(C))+R
sigma_gamma_e = np.concatenate([np.concatenate([sigma_gamma, np.zeros((3, 1))], axis=1), np.concatenate([np.zeros((1, 3)), [[sigma_e]]], axis=1)], axis=0)
print('sigma_gamma_e: ', sigma_gamma_e)
#aa=sigma_gamma_tilde  # need to calculate this
aa = np.eye(3)
bb = -np.dot(np.dot(C, B), np.array(sigma_e))
#print('bb:',bb)
cc = -np.dot(np.dot(sigma_e, np.transpose(B)), np.transpose(C))
dd = sigma_e
sigma_gamma_e_tilde = np.concatenate([np.concatenate( [sigma_gamma_tilde, -np.dot(np.dot(C, B), sigma_e)], axis=1), np.concatenate( [-np.dot(np.dot(sigma_e, np.transpose(B)), np.transpose(C)), [[sigma_e]]], axis=1)], axis=0)
print(sigma_gamma_e_tilde)


# # Attack Model
Ramp_Attack = np.linspace(0, 6, 1200)
e = np.random.normal(0, math.sqrt(sigma_e), N)


# Subroutine for Http communication


async def send_request(target_url, payload, path, timeout=10):
    session_start_time = time.time()
    data = {}
    print(f'payload=', payload)

    async with aiohttp.ClientSession() as session:
        async with session.post(f'http://{target_url}{path}', json=payload, timeout=timeout) as resp:
            try:
                data = await resp.json()
                print(f'data=', data)
            except aiohttp.client_exceptions.ContentTypeError:
                print('Not a json', resp)
            except asyncio.TimeoutError as e:
                print('asyncio.TimeoutError')
            except asyncio.exceptions.TimeoutError as e:
                print('asyncio.exceptions.TimeoutError')

            session_end_time = time.time()
            data['session_start_time'] = session_start_time
            data['duration'] = session_end_time-session_start_time

    return data

'''
Web-server
'''


async def update(request):

    entry = await request.json()
    #entry = request
    print(f'Received Control Information:{entry}')
    #await Q2.put(entry)
    #tasks = set()
    task1 = asyncio.create_task(control_signal(entry))
    await task1
    return web.json_response({'status': 'mpc_1 ok'})


# Control Signal Computation
async def control_signal(request):

    global attacker_pos
    global attacker_ang
    global attacker_speed
    global attacker_beamspeed
    global g


    entry = await request.json()
    print(f'Received State Information: {entry}')

    # Kalman Filter Parameter Initialization
    y_before = np.array([[0], [0]])
    yhat_before = np.array([[0], [0]])
    x_hat_k_k_1 = np.array([[0], [0], [0]])
    x_hat_k_k = np.array([[0], [0], [0]])

    # Controller
    controller = LQ()


    t1 = time.time()

    # # Time loop
    # for i in range(N):

    # atime1 = time.time()

    # sensor_input = request
    # # sensor_input['set_point'] = 0.35*signal.square(0.02 * np.pi  * i*H)
    # # ref.append(0.35*signal.square(0.02 * np.pi  * i*H))
    #
    # #n = SimpleNamespace(**sensor_input)
    # n = {
    #     "pos": sensor_input['pos'],
    #     "speed": sensor_input['speed'],
    #     "ang": sensor_input['ang'],
    #     "beamspeed": sensor_input['beamspeed'],
    #     "set_point": sensor_input['set_point']
    # }

    # print(n)

    # State Variable Extraction
    ang = entry['ang']
    pos = entry['pos']
    set_point = entry['set_point']
    beamspeed = entry['beamspeed']
    speed = entry['speed']
    #g = entry["g"]
    i = entry["i"]

    # Creating SimpleNamespace
    n = SimpleNamespace(ang=float(ang), pos=float(pos), set_point=float(set_point), speed=float(speed),
                            beamspeed=float(beamspeed))

    #### adding noise to measurements
    V = np.random.normal(0., 0.01, (4, 1))  # measurement noise (standard deviation: 0.0316)
    n.pos = n.pos + V[0]
    n.ang = n.ang + V[1]
    n.speed = n.speed + V[2]
    n.beamspeed = n.beamspeed + V[3]


    #Real_pos.append(n.pos)
    real_pos = n.pos
    real_speed = n.speed
    real_ang = n.ang
    real_beamspeed = n.beamspeed
    #Real_speed.append(n.speed)
    #Real_ang.append(n.ang)
    #Real_beamspeed.append(n.beamspeed)

    print('i', i, 'pos=', n.pos, 'ang=', n.ang, 'speed=', n.speed, 'beamspeed=', n.beamspeed)

    # Storing Observation Signal
    if 1100 <= i <= 1950:
        attacker_pos.append(n.pos)
        attacker_ang.append(n.ang)
        attacker_speed.append(n.speed)
        attacker_beamspeed.append(n.beamspeed)
    else:
        pass



    # Replaying Observation Signal
    if 3100 <= i <= 3950:

        n.pos = attacker_pos[(i - 3100)]
        n.ang = attacker_ang[(i - 3100)]
        n.speed = attacker_speed[(i - 3100)]
        n.beamspeed = attacker_beamspeed[(i - 3100)]
        print("n.pos=", n.pos)
        print("n.ang=", n.ang)
        print("n.speed=", n.speed)
    else:
        pass


    #cloud_pos.append(n.pos)
    #cloud_ang.append(n.ang)
    #cloud_speed.append(n.speed)
    #cloud_beamspeed.append(n.beamspeed)
    cloud_pos = n.pos
    cloud_speed = n.speed
    cloud_ang = n.ang
    cloud_beamspeed = n.beamspeed

    #############
    y = np.array([n.pos, n.speed, n.ang])

    print('e: ', e[i])

    # global x_hat_k_k_1
    # global x_hat_k_k
    # global y_before
    # global yhat_before

    # gamma=y_before - yhat_before

    u_before = float(n.beamspeed) / 0.44

    x_hat_k_k_1 = np.dot(A, x_hat_k_k) + np.dot(B, u_before)

    # print('np.dot(A,x_hat_k_k) : ',np.dot(A,x_hat_k_k) )
    # print('np.dot(B,u_before) : ', np.dot(B,u_before) )
    # print('x_hat_k_k_1: ', x_hat_k_k_1)

    gamma = y - np.dot(C, x_hat_k_k_1)

    # print('np.dot(C,x_hat_k_k_1) : ', np.dot(C,x_hat_k_k_1))
    # print('y: ', y)

    x_hat_k_k = x_hat_k_k_1 + np.dot(K, gamma)

    # y_hat = np.dot(C,x_hat)
    # yhat_before=y_hat

    # r=y-y_hat
    # r1=r[0,0]
    # r2=r[1,0]

    # y_before=y

    # a=float(controller.update(n))+e[i]
    # a=controller.uf_prediction(n)

    # global u

    # u=float(a[0])

    print('gamma: ', gamma)

    gamma_ek_vertical = np.array([gamma[0], gamma[1], gamma[2], [e[i - 1]]])

    gamma_ek_Horizonta = np.array([gamma[0, 0], gamma[1, 0], gamma[2, 0], e[i - 1]])

    print('horizon: ', gamma_ek_Horizonta, 'vertical: ', gamma_ek_vertical)

    # sigma_gamma_e_tilde=np.array( [ [aa[0,0], aa[0,1], bb[0,0]], [ aa[1,0], aa[1,1], bb[1,0] ], [cc[0,0], cc[0,1], sigma_e ] ]  )

    Anum_tilde = np.dot(-0.5, np.dot(np.dot(gamma_ek_Horizonta, np.linalg.inv(sigma_gamma_e_tilde)), gamma_ek_vertical))

    Bnum = np.dot(-0.5, np.dot(np.dot(gamma_ek_Horizonta, np.linalg.inv(sigma_gamma_e)), gamma_ek_vertical))

    print('A:', Anum_tilde, 'B: ', Bnum)

    # global g

    # print('try::', math.sqrt(np.linalg.det(sigma_gamma_e)/np.linalg.det(sigma_gamma_e_tilde)))

    # I added abs here to make it positive for sqrt

    # Anum_tilde-Bnum
    print('a-b: ', (Anum_tilde - Bnum))
    # print('Anum_tilde-Bnum: ', max(0, (math.log(math.sqrt(abs(np.linalg.det(sigma_gamma_e)/np.linalg.det(sigma_gamma_e_tilde))))+(Anum_tilde-Bnum))) )

    print(f'added test_statistics term:', math.log(math.sqrt(abs(np.linalg.det(sigma_gamma_e) / np.linalg.det(sigma_gamma_e_tilde)))) + (
                          Anum_tilde - Bnum))
    g = max(0, (g + math.log(math.sqrt(abs(np.linalg.det(sigma_gamma_e) / np.linalg.det(sigma_gamma_e_tilde)))) + (
                    Anum_tilde - Bnum)))
    # ag=math.log(math.sqrt(abs(np.linalg.det(sigma_gamma_e)/np.linalg.det(sigma_gamma_e_tilde))))
    # ag=ag+(Anum_tilde-Bnum)
    # g=ag

    #GG.append(g)

    #ctrl = controller.update(n)

    ctrl = controller.update(n) + e[i]

    t2 = time.time()

    exe_time = t2-t1

    ####################

    if 3100 <= i <= 3950:

        ctrl = ctrl + Ramp_Attack[i - 3100]

    else:
        pass

    ctrl_opt = ctrl.item()

    ctrl_msg = {
        "ctrl": float(ctrl_opt),
        "exe_time": float(exe_time),
        "real_pos": float(real_pos),
        "g": float(g),
        "real_speed": float(real_speed),
        "real_ang": float(real_ang),
        "cloud_pos": float(cloud_pos),
        "cloud_speed": float(cloud_speed),
        "cloud_ang": float(cloud_ang)
    }

    #await send_request('130.235.202.120:54234', ctrl_msg, '/ctrl', 10)

    #task1 = asyncio.create_task(send_request('130.235.202.120:54234', ctrl_msg, '/ctrl', 10))
    #await task1
    # plots()

    return web.json_response(ctrl_msg)





'''
paths
'''
app = web.Application()
# Plant
app.add_routes([
    web.post('/ctrl', control_signal),
    web.post('/update', update)
])

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    print('Starting web client')


    #parser = argparse.ArgumentParser()
    #parser.add_argument('--target', type=str, default='127.0.0.1:8080', help='URL to server')
    #parser.add_argument('--duration', type=int, default=120, help='Duration in seconds')
    #parser.add_argument('--timeout', type=int, default=20, help='Timeouts in seconds')
    #parser.add_argument('--h', type=float, default=0.025, help='Period in seconds')
    #args = parser.parse_args()
    #loop.run_until_complete(main(duration=args.duration, timeout=args.timeout, h=args.h))
    print('Starting web server')
    web.run_app(app, port=MY_PORT)

