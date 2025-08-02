import time
import numpy as np
import cotc.lqr as lqr
import cvxsequence
import math
#import cotc.cloudcontrol as CloudMPC


#import os


class Controller:

    def __init__(self, scale=True):
        self.set_point = 0
        self.scale = scale

    def scaling(self):
        return self.scale

    def update(self):
        raise NotImplementedError


class LQ(Controller):

    def __init__(self, N=20, lqmaxerr=0.01, scale=False):
        super().__init__(scale=scale)

        self.lq = lqr.BnBLQR(20)

    def update(self, request):
        return self.update_param(pos=request.pos, ang=request.ang, speed=request.speed, set_point=request.set_point)

    def update_param(self, pos, ang, speed, set_point):

        u = self.lq.update(x=(pos, speed, ang), xref=(set_point, 0, 0))
        
        return u

import json
from cotc import cvx
from cotc.controller import InfeasibleException
from cotc.opt import LinearConstraints


class MPC_1(Controller):
    
    def __init__(self, scale=False, parameter_file='mpc_parameters.json'):
        super().__init__(scale=scale)

        with open(parameter_file, 'r') as file: 
            parameters = json.load(file)

        # Parameters and constraints
        self.max_iterations = parameters['max_iterations']
        self.N = parameters['N']

        # Initialize and configure MPC
        A = np.array(parameters['model']['A']).reshape(3, 3).T
        B = np.array(parameters['model']['B']).reshape(3, 1)
        Q = np.array(parameters['costs']['Q']).reshape(3, 3).T
        R = np.array(parameters['costs']['R'])
        uc = xc = tc = None
        if 'constraints' in parameters:
          if 'xmin' in parameters['constraints']:
            xc = LinearConstraints.bounds(parameters['constraints']['xmin'], parameters['constraints']['xmax']).get_bounds()
          if 'umin' in parameters['constraints']:
            uc = LinearConstraints.bounds(parameters['constraints']['umin'], parameters['constraints']['umax']).get_bounds()
          if 'xf_min' in parameters['constraints']:
            tc = LinearConstraints.bounds(parameters['constraints']['xf_min'], parameters['constraints']['xf_max']).get_bounds()

        self.mpc = cvxsequence.MPC(A=A, B=B, Q=Q, R=R, horizon=self.N, state_constraints=xc, input_constraints=uc, terminal_constraints=tc)

        self.mpc.set_max_iterations(self.max_iterations)
        self.mpc.set_tolerance(1e-5, 1e-5, 1e-5)

    def update(self, request):
        if request.set_point:
            self.set_point = request.set_point

        try:
          nbr_itr = self.mpc.calc(np.array([request.pos, request.speed, request.ang]), np.array([self.set_point, 0, 0]))
        except InfeasibleException as e:
          print('Oh no')

        return self.mpc.get_u(0)[0]

    def p_rediction(self):
        return self.mpc.u_predictions()


class MPC_2(Controller):

    def __init__(self, scale=False, parameter_file='mpc_parameters.json'):
        super().__init__(scale=scale)

        with open(parameter_file, 'r') as file:
            parameters = json.load(file)

        # Parameters and constraints
        self.max_iterations = parameters['max_iterations']
        self.N = parameters['N']

        # Initialize and configure MPC
        A = np.array(parameters['model']['A']).reshape(3, 3).T
        B = np.array(parameters['model']['B']).reshape(3, 1)
        Q = np.array(parameters['costs']['Q']).reshape(3, 3).T
        R = np.array(parameters['costs']['R'])
        uc = xc = tc = None
        if 'constraints' in parameters:
            if 'xmin' in parameters['constraints']:
                xc = LinearConstraints.bounds(parameters['constraints']['xmin'],
                                              parameters['constraints']['xmax']).get_bounds()
            if 'umin' in parameters['constraints']:
                uc = LinearConstraints.bounds(parameters['constraints']['umin'],
                                              parameters['constraints']['umax']).get_bounds()
            if 'xf_min' in parameters['constraints']:
                tc = LinearConstraints.bounds(parameters['constraints']['xf_min'],
                                              parameters['constraints']['xf_max']).get_bounds()

        self.mpc = cvxsequence.MPC(A=A, B=B, Q=Q, R=R, horizon=math.ceil(self.N/2), state_constraints=xc, input_constraints=uc,
                                   terminal_constraints=tc)

        self.mpc.set_max_iterations(self.max_iterations)
        self.mpc.set_tolerance(1e-5, 1e-5, 1e-5)

    def update(self, request):
        if request.set_point:
            self.set_point = request.set_point

        try:
            nbr_itr = self.mpc.calc(np.array([request.pos, request.speed, request.ang]),
                                    np.array([self.set_point, 0, 0]))
        except InfeasibleException as e:
            print('Oh no')

        return self.mpc.get_u(0)[0]

    def p_rediction(self):
        return self.mpc.u_predictions()


import json
#from cotc import cvxgen as cvxg
from cotc.controller import InfeasibleException
class MPC_CVXGEN(Controller):
    
    def __init__(self, scale=False, parameter_file='mpc_parameters.json'):
        super().__init__(scale=scale)

        with open(parameter_file, 'r') as file: 
            parameters = json.load(file)

        # Parameters and constraints
        self.max_iterations = parameters['max_iterations']
        self.N = parameters['N']

        # Initialize and configure MPC
        self.mpc = cvxg.load_mpc('ballnbeam', self.N)

        self.mpc.set_max_iterations(self.max_iterations)
#        self.mpc.set_costs(N = self.N, **parameters['costs'])
#        self.mpc.set_model(N = self.N, **parameters['model']) 
#        self.mpc.set_constraints( **parameters['constraints'])

    def update(self, request):
        if request.set_point:
            self.set_point = request.set_point

        try:
          nbr_itr = self.mpc.calc(np.array([request.pos, request.speed, request.ang]), np.array([self.set_point, 0, 0]))
        except InfeasibleException as e:
          print('Oh no')

        return self.mpc.get_u(0)[0]


class PID(Controller):

    def __init__(self, scale=True):
        super().__init__(scale=scale)

        print('Set point:{}'.format(self.set_point))
        print('Scale:{}'.format(self.scale))

        self.prev_update = time.time()
        self.prev_pos = 0
        self.prev_ang = 0

    def update(self, request):
        return self.update_params(pos=request.pos, ang=request.ang, set_point=request.set_point)

    def update_params(self, pos, ang, set_point):
        now = time.time()
        dt = now - self.prev_update
        self.prev_update = now

        # Outer: PD
        inner_ref, outer_i = self.pid(y=pos, prev_y=self.prev_pos, y_ref=set_point, dt=dt, td=1.0, ti=5.0, tr=10.0, kp=-0.2, kd=-0.2, n=10.0, beta=1.0)

        # Inner: P
        v, inner_i = self.pid(y=ang, prev_y=self.prev_ang, y_ref=inner_ref, dt=dt, tr=1.0, kp=1.0, beta=1.0)

        self.prev_pos = pos
        self.prev_ang = ang

        #print(' pos:{}, ang:{}, v:{}, set_point:{}/{}'.format(request.pos, request.ang, v, request.set_point, self.set_point))

        return v

    def pid(self, y, prev_y, y_ref, dt, i=0., d=0., td=1., ti=5., tr=10., kp=-.2, ki=0., kd=0., n=10., beta=1.):
        #
        ad = td / (td + n * dt)
        bd = kd * ad * n

        # e
        e = y_ref - y

        # D
        d = ad*d - bd*(y - prev_y)

        # Control signal
        v = kp * (beta * y_ref - y) + i + d

        # I
        i += (ki * dt / ti) * e * (dt / tr) * (y - v)

        return (v, i)
