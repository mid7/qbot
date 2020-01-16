"""
This environment is made for a servo-webcam-RL demo.
There are six servos which control a quadruped with a LED marker.
The agent is rewarded for moving forwards.
"""

import gym
from gym.spaces import Discrete, Tuple, Box
import numpy as np
import cv2
from pyfirmata import Arduino, util
import threading
import time

PINS = [6, 9, 10, 11]
A_MIN = 0.65
A_MAX = 0.85
DELAY_TIME = 0.25  # 0.5
N_MOTORS = len(PINS)
N_ACTIONS = 2  # 3
N_STATES = 9  # 5
STEP_LIMIT = 5 * 60 / DELAY_TIME
STEP_SIZE = (A_MAX - A_MIN) / N_STATES
DOMAIN = "continuous"  # "discrete"  #


class Env3(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):

        # Initialize environment
        if DOMAIN == "discrete":
            self.action_space = Tuple([Discrete(N_ACTIONS) for n in range(N_MOTORS)])
            self.observation_space = Tuple([Discrete(N_STATES) for n in range(N_MOTORS)])
        elif DOMAIN == "continuous":
            self.action_space = Discrete(N_ACTIONS**N_MOTORS)
            self.observation_space = Box(low=0, high=N_STATES, shape=(N_MOTORS, ), dtype=int)
        else:
            raise Exception
        self.step_no = 0

        # Initialize Arduino
        self.board = Arduino("/dev/ttyACM0")
        print("Connected to Arduino!")
        self.motor = [self.board.get_pin('d:{}:p'.format(pin)) for pin in PINS]
        self.state = np.zeros(N_MOTORS).astype(int) + N_STATES // 2
        self.move()

        # Start camera stream
        self.camera = cv2.VideoCapture(1)
        self.new_position = None
        self.old_position = None
        t = threading.Thread(target=self.stream)
        t.daemon = True
        t.start()
        print("Beginning camera stream...")

        self.reset()

    def stream(self):
        self.new_position = np.array([0, 0])
        self.old_position = np.array([0, 0])
        while True:
            for i in range(5):
                self.camera.grab()
            _, frame = self.camera.read()

            # Define lower and upper limits for object threshold
            lower_red = np.array([100, 100, 100])
            upper_red = np.array([255, 255, 255])

            # Mask the image using the boundaries
            mask = cv2.inRange(frame, lower_red, upper_red)
            res = cv2.bitwise_and(frame, frame, mask=mask)

            # Calculate moments of binary image and x,y coordinate of center
            mom = cv2.moments(mask)

            try:
                c_x = int(mom["m10"] / mom["m00"])
                c_y = int(mom["m01"] / mom["m00"])
            except Exception:
                c_x = self.new_position[0]
                c_y = self.new_position[1]

            # Show text and highlight the center of the circle
            cv2.circle(res, (c_x, c_y), 5, (255, 255, 255), -1)
            cv2.putText(
                res,
                "{}, {}".format(str(c_x), str(c_y)),
                (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.imshow('frame', frame)
            cv2.imshow('mask', mask)
            cv2.imshow('res', res)
            cv2.waitKey(1)

            self.new_position = np.array([c_x, c_y])

    def move(self):
        self.state = np.clip(self.state, 0, N_STATES - 1)
        angle = self.state * STEP_SIZE + A_MIN
        for i in range(N_MOTORS):
            self.motor[i].write(angle[i])
        time.sleep(DELAY_TIME)

    def step(self, action):
        # Move motors
        self.step_no += 1
        if DOMAIN == "continuous":
            padding = "0{}b".format(N_MOTORS)
            action = format(action, padding)
            action = list(action)
            action = np.asarray(action).astype(int)
        for i in range(N_MOTORS):
            if N_ACTIONS == 3:
                self.state[i] += action[i] - 1
            elif N_ACTIONS == 2:
                self.state[i] += 3 ** action[i] - 2
            else:
                raise Exception

        self.move()

        # Calculate reward
        penalty = 0
        for x in self.state:
            if x == N_STATES - 1 or x == 0:
                penalty += 1
        reward = self.new_position[0] - self.old_position[0] - penalty
        self.old_position = self.new_position
        print(reward)

        return tuple(self.state,), reward, not self.step_no % STEP_LIMIT, {}

    def reset(self):
        input("Press Enter to set angles to min")
        self.state = np.zeros(N_MOTORS).astype(int)
        self.move()
        input("Press Enter to set angles to max")
        self.state = np.zeros(N_MOTORS).astype(int) + N_STATES
        self.move()
        input("Press Enter to set angles to mid")
        self.state = np.zeros(N_MOTORS).astype(int) + N_STATES // 2
        self.move()
        input("Press Enter to train")
        self.old_position = self.new_position
        return tuple(self.state,)

    def render(self, mode='human'):
        pass

    def close(self):
        pass

