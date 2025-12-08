import cv2 as cv
import torch
# import os
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys
import socket
import subprocess
import time
import math

def getOdo(socket):
    idxx = -1
    idxy = -1
    idxth = -1

    cmd = 'eval $odox ; $odoy ; $odoth\r\n'
    socket.sendall(cmd.encode())


    while idxx < 0:
        print("recive data")
        data = socket.recv(1024)
        data = data.decode()  # Convert bytes to strin
        # data = repr(data)
        print(data)

    #     idxx = data.find('odox')
    #     idxy = data.find('odoy')
    #     idxth = data.find('odoth')

    # print('init:odox:', odox,'odoy:', odoy, 'odoth:', odoth)

    # return float(data[idxx + 7:idxx + 15]), float(data[idxy + 7:idxy + 15]), float(data[idxth + 7:idxth + 15])
        # data = data.decode()  # Convert bytes to string
        numbers = data.strip()[:].split()  # Remove leading/trailing whitespace and exclude the newline character
        # numbers = data.strip().split()  # Remove leading/trailing whitespace and split the string by spaces
        print(numbers)
        # Extract float numbers
        if numbers[-1] != 'queued' :
            float_numbers = [float(num) for num in numbers[-3:]]
            idxx = 1
            odox, odoy, odoth = float_numbers

    print('init:odox:', odox, 'odoy:', odoy, 'odoth:', odoth)

    return odox, odoy, odoth

if __name__ == "__main__":

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('192.168.2.2', 31001)
    s.connect(server_address)
    time.sleep(4)

    cmd = 'set "$odox" 0\r\n'
    s.send(cmd.encode())
    cmd = 'set "$odoy" 0\r\n'
    s.send(cmd.encode())
    cmd = 'set "$odoth" 0\r\n'
    s.send(cmd.encode())
    print("Done.")
    time.sleep(1)
    # Setup the robot
    # cmd = 'set "xmlon" 1\r\n'
    # s.send(cmd.encode())


    debug = True

    odox, odoy, odoth = getOdo(s)
    odox = odox + 1.5
    odoy = odoy - 1

    flag = 0

    while debug:
        # flush the current commands on the robot
        cmd = "flushcmds\r\n"
        s.send(cmd.encode())

        vel = 0.5

        # Wait for a key press and close the window
        key = cv.waitKey(1)
        if key & 0xFF == ord('q') or flag == 30:
            cmd = "idle\r\n"
            s.send(cmd.encode())
            # time.sleep(2)
            debug = False

        else:
            th = np.rad2deg(odoth)
            cmd = "driveon " + str(odox) + " " + str(odoy) + " " + str(round(th, 2)) + " @v" + str(
                vel) + ":($cmdtime>10)\r\n"
            print('odox:', odox,'odoy:', odoy, 'odoth:', odoth)
            # s.send(cmd.encode())

        flag += 1
        time.sleep(1)

    odox, odoy, odoth = getOdo(s)
    print('odox:', odox,'odoy:', odoy, 'odoth:', odoth)
