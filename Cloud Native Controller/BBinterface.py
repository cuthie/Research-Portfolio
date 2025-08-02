class BBInterface():

    def read_ang(self):
        raise NotImplementedError

    def read_pos(self):
        raise NotImplementedError

    def read_speed(self):
        raise NotImplementedError

    def act(self, value):
        raise NotImplementedError

    def on_terminate(self):
        raise NotImplementedError

    def read(self):
        raise NotImplementedError


import cotc.sim.ballnbeam as bbsim
class BBsimEulerInterface(BBInterface):

    def __init__(self, scale=True, h=0.05):
        self.scale = scale
        self.h = h
        self.bb = bbsim.BBSimInterfaceEuler()

    def read_ang(self):
        pass

    def read_pos(self):
        pass

    def read_speed(self):
        pass

    def act(self, value):
        self.bb.set_beam_speed(value)
        #self.bb.step(self.h)
        
    def on_terminate(self):
        pass 
####
    def nextstep(self):
        self.bb.step(self.h)


    #def intState(self, ballposition, ballspeed, beamangle):
     #   self.bb.set_state(ballposition, ballspeed, beamangle)

####    
    def read(self):
        # [pos, speed, ang]
        x = self.bb.get_state()
        return {'pos':x[0], 'speed':x[1], 'ang':x[2], 'beamspeed':x[3]}

import math
import sys

class BBsimShmInterface(BBInterface):

    def __init__(self, scale=True):
        self.scale = scale
        self.bb = bbsim.BBSimInterfacePosixShm()

    def read_ang(self):
        scale = 0.785
        value = self.bb.get_angle()

        if self.scale:
            value = value/scale*10.0
            value = max(-10.0, min(10.0, value))

        return value

    def read_pos(self):
        scale = 0.55

        value = self.bb.get_position()
        
        if self.scale:
            value = value/scale*10.0
            value = max(-10.0, min(10.0, value))
            
        return value

    def read_speed(self):
        return self.bb.get_ball_speed()

    def act(self, value):
        #if self.scale:
        #    value = (value/10.0)*2*math.pi
        self.bb.set_beam_speed(value)

    def on_terminate(self):
        self.bb.reset()

import posix_ipc as ipc 
import math, time

class BBsimInterface(BBInterface):

    def __init__(self, scale=True, id=0):
        self.scale = scale

        self.ACT_DEV = '/bbsim_in-{}'.format(id)
        self.ANG_DEV = '/bbsim_out1-{}'.format(id)
        self.POS_DEV = '/bbsim_out2-{}'.format(id)
        self.SPEED_DEV = '/bbsim_ballspeed-{}'.format(id)

        self.act_queue = ipc.MessageQueue(self.ACT_DEV, max_messages=1)
        self.ang_queue = ipc.MessageQueue(self.ANG_DEV, max_messages=1)
        self.pos_queue = ipc.MessageQueue(self.POS_DEV, max_messages=1)
        self.speed_queue = ipc.MessageQueue(self.SPEED_DEV, max_messages=1)
        self._reset = ipc.MessageQueue("/bbsim_reset-{}".format(id), max_messages=1)   

        self.reset()

    def reset(self):
        try:
            self._reset.send("{}".format(1))
        except ipc.BusyError:
            print("Failed to send reset, check your code, this ought to not happen")

    def read_ang(self):
        scale = 0.785
        value = None
        try:
            message, priority = self.ang_queue.receive(1) # Get input signal
            value = float(message)

            if self.scale:
                value = value/scale*10.0
                value = max(-10.0, min(10.0, value))

        except ipc.BusyError as err:
            print('blä - ang - {}'.format(err))
        return value

    def read_pos(self):
        scale = 0.55
        value = None
        try:
            message, priority = self.pos_queue.receive(1) # Get input signal
            value = float(message)
        
            if self.scale:
                value = value/scale*10.0
                value = max(-10.0, min(10.0, value))
            
        except ipc.BusyError as err:
            print('blä - pos - {}'.format(err))
        return value

    def read_speed(self):
        value = None
        try:
            message, priority = self.pos_queue.receive(1) # Get input signal
            value = float(message)
        except ipc.BusyError as err:
            print('blä - speed - {}'.format(err))
        return value

    def act(self, value):
        try:
            #self.act_queue.send("{}".format((value/10.0)*2*math.pi), 0)
            self.act_queue.send(value, 0)
        except ipc.BusyError:
            print("Failed to set new input, this should not happen\n")
    
    def on_terminate(self):
        self.reset()
        self.act_queue.close()
        self.ang_queue.close()
        self.pos_queue.close()
        self.speed_queue.close()