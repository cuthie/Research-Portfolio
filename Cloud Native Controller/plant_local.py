import asyncio
import aiohttp
from aiohttp import web
import time
import argparse
from BBinterface import BBsimEulerInterface
from types import SimpleNamespace
from Controller import LQ
from asyncio import Queue
import socket
import sys
import os
import matplotlib.pyplot as plt




# Assigning a port range
PORT_RANGE_MIN = 49152 
PORT_RANGE_MAX = 65535
MY_PORT = 54234
#random.randint(PORT_RANGE_MIN, PORT_RANGE_MAX)

# Control queues
ctrl_queue = Queue()

# Timeouts 
REGISTER_TIMEOUT = 20  # Seconds

# Events 
ready_event = asyncio.Event()

# Parameters
H = 0.05
SET_POINT = 0.3

# Maximum number of iteration
p_max = 100

N = 3950

# The plant
plant = BBsimEulerInterface(h=H)

# # Control message
# ctrl_opt = {}

'''
Function that sends and processes requests
'''


async def send_request(target_url, payload, path, timeout=10):
    session_start_time = time.time()
    data = {}

    async with aiohttp.ClientSession() as session:
        async with session.post(f'http://{target_url}{path}', json=payload, timeout=timeout) as resp:
            try:
                data = await resp.json()
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
Server functions
'''
# Receiving control messages and puts them in the control queue


async def control_signal(request):

    entry = await request.json()
    print(f'Received ctrl: {entry}')
    await ctrl_queue.put(entry)
    return web.json_response({'status': 'ok'})

# Management


async def ping(request):
    return web.json_response({'time': time.time()})


# Define webserver
app = web.Application()
app.add_routes([
    web.post('/ctrl', control_signal)
    ])

'''
The main loop
'''


async def main(duration, timeout, h):
    ''' Start web server for inbound control messages '''
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', MY_PORT)
    await site.start()

    ''' The control loop '''
    # Variables and time-keeping
    current_time = time.time()

    end_time = current_time + duration
    next_session_start = current_time

    # Pending controller tasks
    pending_tasks = set()

    # Loop counter variable
    p = 0
    GG = []
    Real_pos = []
    Real_speed = []
    Real_ang = []
    #g = 0

    Cloud_pos = []
    Cloud_speed = []
    Cloud_ang = []

    # # The actual loop
    # while time.time() <= end_time:  # Run for duration
    #
    #     # Time keeping
    #     next_session_start += h

    ''' Sampling the plant '''

    for i in range(N):

        # Read the values from the plant
        sensor_input = plant.read()

        #print(sensor_input)

        # Adding the Set point to the dictionary
        sensor_input['set_point'] = SET_POINT

        #print(sensor_input)

        #n = SimpleNamespace(**sensor_input)

        # Creating a json object for the state variable
        n = {
            "pos": float(sensor_input['pos']),
            "speed": float(sensor_input['speed']),
            "ang": float(sensor_input['ang']),
            "beamspeed": float(sensor_input['beamspeed']),
            "set_point": float(sensor_input['set_point']),
            "i": int(i)
        }

        # Increasing the counter variable
        #p += 1

        print(n)

        # # State Variable Extraction
        # ang = sensor_input['ang']
        # pos = sensor_input['pos']
        # set_point = sensor_input['set_point']
        # beamspeed = sensor_input['beamspeed']
        # speed = sensor_input['speed']
        #
        # req = SimpleNamespace(ang=float(ang), pos=float(pos), set_point=float(set_point), speed=float(speed), beamspeed=float(beamspeed))

        # The time counter
        t1 = time.time()

        # Get control messages
        try:

            # Request controllers
            pending_tasks.add(asyncio.create_task(send_request('127.0.0.1:60031', n, '/ctrl', 10)))

            # Wait for response for remaining time until next time slot
            diff = next_session_start - time.monotonic()
            (done_tasks, pending_tasks) = await asyncio.wait(
                                                pending_tasks,
                                                return_when=asyncio.ALL_COMPLETED,
                                                timeout=max(0, diff)
                                                )

            ctrl_opt = {}

            # Check for the completed tasks
            for task in done_tasks:
                ctrl_opt = task.result()

            #ctrl_queue.get(), timeout=max(next_session_start - time.time(), 0)# Wait at the most until the next session starts

            #ctrl_opt = await ctrl_queue.get_nowait()

            # try:
            #     ctrl_opt = await ctrl_queue.get_nowait()
            # except asyncio.queues.QueueEmpty:
            #     ctrl_opt = await ctrl_queue.get()

            #print(ctrl_opt)

            # Control Signal
            ctrl = ctrl_opt["ctrl"]

            # The time for the process
            exe_time = ctrl_opt["exe_time"]

            # The Test Statistics
            g = ctrl_opt["g"]
            GG.append(g)

            # The Real Position
            real_pos = ctrl_opt["real_pos"]
            Real_pos.append(real_pos)

            # The Real Speed
            real_speed = ctrl_opt["real_speed"]
            Real_speed.append(real_speed)

            # The Real Angle
            real_ang = ctrl_opt["real_ang"]
            Real_ang.append(real_ang)

            # The Cloud Position
            cloud_pos = ctrl_opt["cloud_pos"]
            Cloud_pos.append(cloud_pos)

            # The Cloud Speed
            cloud_speed = ctrl_opt["cloud_speed"]
            Cloud_speed.append(cloud_speed)

            # The Cloud Angle
            cloud_ang = ctrl_opt["cloud_ang"]
            Cloud_ang.append(cloud_ang)

            x = {
                "real_pos": float(real_pos),
                "real_speed": float(real_speed),
                "real_ang": float(real_ang),
            }

            y = {
                "cloud_pos": float(cloud_pos),
                "cloud_speed": float(cloud_speed),
                "cloud_ang": float(cloud_ang),
            }

            print(x)
            print(y)


            # Actuation on plant
            plant.act(value=ctrl)
            plant.nextstep()

            # Printing the execution time
            print(exe_time)


        except asyncio.TimeoutError:
            print('No control messages from the cloud')

        # # The time counter
        # t2 = time.time()
        #
        # # Elapsed time
        # print(t2-t1)
        #
        # # Sleep for remainder of the time
        # time.sleep(max(next_session_start - time.time(), 0))

    # print("GG=", GG)
    # print("Real_pos=", Real_pos)
    # print("Real_speed=", Real_speed)
    # print("Real_ang=", Real_ang)
    # print("cloud_pos=", Cloud_pos)
    # print("cloud_speed=", Cloud_speed)
    # print("cloud_ang=", Cloud_ang)

    # # Implementation Plots
    # plt.figure()
    # plt.plot(GG)
    # plt.title('g')

    plt.figure()
    plt.xlabel('Sample Instances')
    plt.ylabel('Magnitude')
    #plt.plot(Real_pos, color='r', label='Real Position')
    #plt.title('real pos')

    # plt.figure()
    plt.plot(Real_ang, color='g', label='Real Angle')
    #plt.title('real speed')

    # plt.figure()
    # plt.plot(Real_ang)
    # plt.title('real angle')
    #
    # plt.figure()
    plt.plot(Cloud_ang, color='m', label='Cloud Angle')
    #plt.title('cloud pos')
    #
    # plt.figure()
    # plt.plot(Cloud_speed)
    # plt.title('cloud speed')
    #
    # plt.figure()
    # plt.plot(Cloud_ang)
    # plt.title('cloud angle')

    plt.legend()

    plt.show()

    await runner.cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--target', type=str, default='127.0.0.1:8080', help='URL to server')
    parser.add_argument('--duration', type=int, default=120, help='Duration in seconds')
    parser.add_argument('--timeout', type=int, default=20, help='Timeouts in seconds')
    parser.add_argument('--h', type=float, default=0.025, help='Period in seconds')

    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(duration=args.duration, timeout=args.timeout, h=args.h))
