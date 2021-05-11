import logging
import math

import gym
import numpy as np

from gym import make as gym_make
from gym import spaces
from gym.utils import seeding
from math import  *
logger = logging.getLogger(__name__)
import gym
import numpy as np
from math import *
from numpy.linalg import inv
import torch
from typing import NamedTuple
# import config
import os
import pygame

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

class CartPoleSwingUpContinuousEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.g = 9.82  # gravity
        self.m_c = 0.5  # cart mass
        self.m_p = 0.5  # pendulum mass
        self.total_m = (self.m_p + self.m_c)
        self.l = 0.6  # pole's length
        self.m_p_l = (self.m_p * self.l)
        self.force_mag = 10.0
        self.dt = 0.01  # seconds between state updates
        self.b = 0.1  # friction coefficient

        self.t = 0  # timestep
        self.t_limit = 500

        # Angle, angle speed and speed at which to fail the episode
        self.x_threshold = 6
        self.x_dot_threshold = 10
        self.theta_dot_threshold = 10

        high = np.array([
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max])

        self.action_space = spaces.Box(-10.0, 10.0, shape=(1,))
        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # valid action
        #PDP
        # action = np.clip(action, -10.0, 10.0)[0]
        action = action[0]

        state = self.state
        x, x_dot, theta, theta_dot = state

        # nullify action, if it would normally push the cart out of boundaries
        if x >= self.x_threshold and action >= 10:
            action = 0
        elif x <= -self.x_threshold and action <= -10:
            action = 0

        s = math.sin(theta)
        c = math.cos(theta)


        # xdot_update = (-2 * self.m_p_l * (
        #         theta_dot ** 2) * s + 3 * self.m_p * self.g * s * c + 4 * action - 4 * self.b * x_dot) / (
        #                       4 * self.total_m - 3 * self.m_p * c ** 2)
        # thetadot_update = (-3 * self.m_p_l * (theta_dot ** 2) * s * c + 6 * self.total_m * self.g * s + 6 * (
        #         action - self.b * x_dot) * c) / (4 * self.l * self.total_m - 3 * self.m_p_l * c ** 2)

        #pdp
        x, dx,  q, dq = x, x_dot, theta, theta_dot
        mp = self.m_p
        mc =  self.m_c
        l = self.l
        g= self.g
        xdot_update = (action + mp * sin(q) * (l * dq * dq + g * cos(q))) / (
                mc + mp * sin(q) * sin(q))  # acceleration of x
        thetadot_update = (-action * cos(q) - mp * l * dq * dq * sin(q) * cos(q) - (
                mc + mp) * g * sin(
            q)) / (
                        l * mc + l * mp * sin(q) * sin(q))  # acceleration of theta


        x = x + x_dot * self.dt
        theta = theta + theta_dot * self.dt
        x_dot = x_dot + xdot_update * self.dt
        theta_dot = theta_dot + thetadot_update * self.dt




        self.state = (x, x_dot, theta, theta_dot)

        done = False

        # restrict state of cart to be within its limits without terminating the game
        if x > self.x_threshold:
            x = self.x_threshold
        elif x < -self.x_threshold:
            x = -self.x_threshold
        elif x_dot > self.x_dot_threshold:
            x_dot = self.x_dot_threshold
        elif x_dot < -self.x_dot_threshold:
            x_dot = -self.x_dot_threshold
        elif theta_dot > self.theta_dot_threshold:
            theta_dot = self.theta_dot_threshold
        elif theta_dot < -self.theta_dot_threshold:
            theta_dot = -self.theta_dot_threshold

        self.t += 1

        # terminate the game if t >= time limit
        if self.t >= self.t_limit:
            done = True

        # reward function as described in dissertation of Deisenroth with A=1
        A = 1
        invT = A * np.array([[1, self.l, 0], [self.l, self.l ** 2, 0], [0, 0, self.l ** 2]])
        j = np.array([x, np.sin(theta), np.cos(theta)])
        #PDP
        j_target = np.array([0.0, 0.0, -1.0])

        reward = np.matmul((j - j_target), invT)
        reward = np.matmul(reward, (j - j_target))
        reward = -(1 - np.exp(-0.5 * reward))

        obs = np.array([x, x_dot, theta, theta_dot])

        return obs, reward, done, {}

    def reset(self):
        # spawn cart at the same initial state + randomness
        # self.state = np.random.normal(loc=np.array([0.0, 0.0, np.pi, 0.0]), scale=np.array([0.01, 0.01, 0.01, 0.01]))
        #PDP
        self.state = np.random.normal(loc=np.array([0.0, 0.0, 0.0, 0.0]), scale=np.array([0.01, 0.01, 0.01, 0.01]))
        self.t = 0  # timestep
        obs = self.state
        return obs

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 600

        world_width = 5
        scale = screen_width / world_width
        carty = screen_height / 2
        polewidth = 6.0
        polelen = scale * self.l
        cartwidth = 40.0
        cartheight = 20.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2

            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            cart.set_color(1, 0, 0)
            self.viewer.add_geom(cart)

            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0, 0, 1)
            self.poletrans = rendering.Transform(translation=(0, 0))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)

            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.1, 1, 1)
            self.viewer.add_geom(self.axle)

            self.pole_bob = rendering.make_circle(polewidth / 2)
            self.pole_bob_trans = rendering.Transform()
            self.pole_bob.add_attr(self.pole_bob_trans)
            self.pole_bob.add_attr(self.poletrans)
            self.pole_bob.add_attr(self.carttrans)
            self.pole_bob.set_color(0, 0, 0)
            self.viewer.add_geom(self.pole_bob)

            self.wheel_l = rendering.make_circle(cartheight / 4)
            self.wheel_r = rendering.make_circle(cartheight / 4)
            self.wheeltrans_l = rendering.Transform(translation=(-cartwidth / 2, -cartheight / 2))
            self.wheeltrans_r = rendering.Transform(translation=(cartwidth / 2, -cartheight / 2))
            self.wheel_l.add_attr(self.wheeltrans_l)
            self.wheel_l.add_attr(self.carttrans)
            self.wheel_r.add_attr(self.wheeltrans_r)
            self.wheel_r.add_attr(self.carttrans)
            self.wheel_l.set_color(0, 0, 0)
            self.wheel_r.set_color(0, 0, 0)
            self.viewer.add_geom(self.wheel_l)
            self.viewer.add_geom(self.wheel_r)

            self.track = rendering.Line(
                (screen_width / 2 - self.x_threshold * scale, carty - cartheight / 2 - cartheight / 4),
                (screen_width / 2 + self.x_threshold * scale, carty - cartheight / 2 - cartheight / 4))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(x[2]-np.pi)
        self.pole_bob_trans.set_translation(-self.l * np.sin(x[2]-np.pi), self.l * np.cos(x[2]-np.pi))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
    
    def np_dynamics(self,state,action):
        x, x_dot, theta, theta_dot = state
        action = action[0]

        # nullify action, if it would normally push the cart out of boundaries
        if x >= self.x_threshold and action >= 10:
            action = 0
        elif x <= -self.x_threshold and action <= -10:
            action = 0

        #pdp
        x, dx,  q, dq = x, x_dot, theta, theta_dot
        mp = self.m_p
        mc =  self.m_c
        l = self.l
        g= self.g
        xdot_update = (action + mp * sin(q) * (l * dq * dq + g * cos(q))) / (
                mc + mp * sin(q) * sin(q))  # acceleration of x
        thetadot_update = (-action * cos(q) - mp * l * dq * dq * sin(q) * cos(q) - (
                mc + mp) * g * sin(
            q)) / (l * mc + l * mp * sin(q) * sin(q))  # acceleration of theta
        x = x + x_dot * self.dt
        theta = theta + theta_dot * self.dt
        x_dot = x_dot + xdot_update * self.dt
        theta_dot = theta_dot + thetadot_update * self.dt

        # restrict state of cart to be within its limits without terminating the game
        if x > self.x_threshold:
            x = self.x_threshold
        elif x < -self.x_threshold:
            x = -self.x_threshold
        elif x_dot > self.x_dot_threshold:
            x_dot = self.x_dot_threshold
        elif x_dot < -self.x_dot_threshold:
            x_dot = -self.x_dot_threshold
        elif theta_dot > self.theta_dot_threshold:
            theta_dot = self.theta_dot_threshold
        elif theta_dot < -self.theta_dot_threshold:
            theta_dot = -self.theta_dot_threshold
        

        return np.array([x, x_dot, theta, theta_dot])





class EnvSpec(NamedTuple):
    observation_space: int
    action_space: int

class RobotArm(gym.Env):
    def __init__(self):
        high = np.array([np.pi,np.pi,np.pi,np.pi],
            dtype=np.float32,)
        self.action_space = gym.spaces.Box(low=-1., high=1., shape=(2,))
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.state_size, self.action_size = 4, 2
        self.observation_size = self.state_size

        self.dt = 0.1
        self.g =0.0
        self.x_goal = [pi / 2, 0, 0, 0]
        self.wq1, self.wq2, self.wdq1, self.wdq2, self.wu = 0.1, 0.1, 0.1, 0.1, 0.01
        self.l1,self.m1,self.l2,self.m2 = 1.0,1.0,1.0,1.0
        # self.true_params = torch.tensor([self.l1,self.m1,self.l2,self.m2]).to(config.device) #l1,m1,l2,m2 
        self.unwrapped.reward =  self.reward_func
        self.env_spec = EnvSpec(self.observation_space.shape[0], self.action_space.shape[0])
        self.init_window()


    def dynamics_true(self,state,action):
        # set variable
        q1, q2, dq1,dq2 = state
        u1, u2 = action
        l1,m1,l2,m2 = self.l1,self.m1,self.l2,self.m2 
        g=self.g
        dt=self.dt

        # Declare model equations (discrete-time)
        r1 = l1 / 2
        r2 = l2 / 2
        I1 = l1 * l1 * m1 / 12
        I2 = l2 * l2 * m2 / 12
        M11 = m1 * r1 * r1 + I1 + m2 * (l1 * l1 + r2 * r2 + 2 * l1 * r2 * cos(q2)) + I2
        M12 = m2 * (r2 * r2 + l1 * r2 * cos(q2)) + I2
        M21 = M12
        M22 = m2 * r2 * r2 + I2
        M = np.vstack((np.hstack((M11,M12)),np.hstack((M21,M22))))

        h = m2 * l1 * r2 * sin(q2)
        C1 = -h * dq2 * dq2 - 2 * h * dq1 * dq2
        C2 = h * dq1 * dq1
        C = np.vstack((C1, C2))

        G1 = m1 * r1 * g * cos(q1) + m2 * g * (r2 * cos(q1 + q2) + l1 * cos(q1))
        G2 = m2 * g * r2 * cos(q1 + q2)
        G = np.vstack((G1, G2))
        
        # ddq = mtimes(inv(M), -C - G + self.U)  # joint acceleration
        U = np.vstack((u1, u2))
        ddq = inv(M) @ (-C-G+U)
        f = np.hstack((dq1,dq2,ddq[0],ddq[1]))
        new_state = state + dt*f
        new_state[0] = angle_normalize(new_state[0])
        new_state[1] = angle_normalize(new_state[1])
        return new_state

    def step(self,action):
        state = self.state
        newstate = self.dynamics_true(state,action)
        reward = self.reward_func(state,action)
        done = False
        self.state = newstate
        state = newstate

        return state, reward, done, {}
    
    def reset(self):
        self.state = np.zeros((self.observation_space.shape[0],))
        done = False
        state =  self.state
        return state
    

    def reward_func(self,state,action):
        """
        torch
        """
        q1, q2, dq1,dq2 = state[0],state[1],state[2],state[3]
        cost_q1 = (q1 - self.x_goal[0]) ** 2
        cost_q2 = (q2 - self.x_goal[1]) ** 2
        cost_dq1 = (dq1 - self.x_goal[2]) ** 2
        cost_dq2 = (dq2 - self.x_goal[3]) ** 2
        cost_u = np.sum(action**2)

        path_cost = self.wq1 * cost_q1 + self.wq2 * cost_q2 + \
                         self.wdq1 * cost_dq1 + self.wdq2 * cost_dq2 + self.wu * cost_u
        reward = -1 * path_cost
        return reward
    
    def init_window(self):
        os.environ['SDL_AUDIODRIVER'] = 'dsp'
        self.window_size = [600,600]
        self.centre_window = [self.window_size[0]//2, self.window_size[1]//2]
        self.set_link_properties([100,100])
        self.viewer = None

    def rotate_z(self, theta):
        rz = np.array([[np.cos(theta), - np.sin(theta), 0, 0],
                       [np.sin(theta), np.cos(theta), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
        return rz

    def translate(self, dx, dy, dz):
        t = np.array([[1, 0, 0, dx],
                      [0, 1, 0, dy],
                      [0, 0, 1, dz],
                      [0, 0, 0, 1]])
        return t

    def forward_kinematics(self, theta):
        P = []
        P.append(np.eye(4))
        for i in range(0, self.n_links):
            R = self.rotate_z(theta[i])
            T = self.translate(self.links[i], 0, 0)
            P.append(P[-1].dot(R).dot(T))
        return P

    def inverse_theta(self, theta):
        new_theta = theta.copy()
        for i in range(theta.shape[0]):
            new_theta[i] = -1*theta[i]
        return new_theta

    def set_link_properties(self, links):
        self.links = links
        self.n_links = len(self.links)
        self.min_theta = math.radians(0)
        self.max_theta = math.radians(90)
        self.max_length = sum(self.links)

    def draw_arm(self, theta):
        LINK_COLOR = (255, 255, 255)
        JOINT_COLOR = (0, 0, 0)
        TIP_COLOR = (0, 0, 255)
        theta = self.inverse_theta(theta)
        P = self.forward_kinematics(theta)
        origin = np.eye(4)
        origin_to_base = self.translate(self.centre_window[0],self.centre_window[1],0)
        base = origin.dot(origin_to_base)
        F_prev = base.copy()
        for i in range(1, len(P)):
            F_next = base.dot(P[i])
            pygame.draw.line(self.screen, LINK_COLOR, (int(F_prev[0,3]), int(F_prev[1,3])), (int(F_next[0,3]), int(F_next[1,3])), 5)
            pygame.draw.circle(self.screen, JOINT_COLOR, (int(F_prev[0,3]), int(F_prev[1,3])), 10)
            F_prev = F_next.copy()
        pygame.draw.circle(self.screen, TIP_COLOR, (int(F_next[0,3]), int(F_next[1,3])), 8)

    def render(self, mode='human'):
        SCREEN_COLOR = (50, 168, 52)
        if self.viewer == None:
            pygame.init()
            pygame.display.set_caption("RobotArm-Env")
            self.screen = pygame.display.set_mode(self.window_size)
            self.clock = pygame.time.Clock()
        self.screen.fill(SCREEN_COLOR)
        theta = np.array([self.state[0],self.state[1]])
        self.draw_arm(theta)
        self.clock.tick(60)
        pygame.display.flip()
