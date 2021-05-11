#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 12:41:57 2018
@author: matthewszhang
"""

# -----------------------------------------------------------------------------
#   @author:
#       Matthew Zhang
#   @brief:
#       Several basic classical control environments that
#       1. Provide ground-truth reward function.
#       2. Has reward as a function of the observation.
#       3. has episodes with fixed length.
# -----------------------------------------------------------------------------
import numpy as np

from mbbl.config import init_path
from mbbl.env import base_env_wrapper as bew
from mbbl.env import env_register
from mbbl.env import env_util
from mbbl.util.common import logger


class env(bew.base_env):
    # acrobot has applied sin/cos obs
    SWINGUP = ['gym_robotarm']

    def __init__(self, env_name, rand_seed, misc_info):
        super(env, self).__init__(env_name, rand_seed, misc_info)
        self._base_path = init_path.get_abs_base_dir()

        # return the reset as the gym?
        if 'reset_type' in misc_info and misc_info['reset_type'] == 'gym':
            self._reset_return_obs_only = True
            self.observation_space, self.action_space = \
                self._env.observation_space, self._env.action_space
            # it's possible some environments have different obs
            self.observation_space = \
                env_util.box(self._env_info['ob_size'], -np.pi,np.pi)

            # note: we always set the action to range with in [-1, 1]
            self.action_space.low[:] = -1.0
            self.action_space.high[:] = 1.0
        else:
            self._reset_return_obs_only = False
        
        self._env.env.max_torque = 1.0

    def step(self, action):
        #TODO: add force amag
        true_action = action * self._env.env.max_torque
        _, _r, _, info = self._env.step(true_action)
        ob = self._get_observation()

        # get the reward
        reward = self.reward(
            {'end_state': ob, 'start_state': self._old_ob, 'action': action,"reward":_r}
        )

        # get the end signal
        self._current_step += 1
        if self._current_step > self._env_info['max_length']:
            done = True
        else:
            done = False  # will raise warnings -> set logger flag to ignore
        self._old_ob = np.array(ob)
        return ob, reward, done, info

    """
    def reset(self, control_info={}):
        self._current_step = 0
        self._old_ob = self._env.reset()
        return np.array(self._old_ob), 0.0, False, {}
    """

    def reset(self, control_info={}):
        self._current_step = 0
        self._env.reset()

        # the following is a hack, there is some precision issue in mujoco_py
        self._old_ob = self._get_observation()
        self._env.reset()
        self.set_state({'start_state': self._old_ob.copy()})
        self._old_ob = self._get_observation()

        if self._reset_return_obs_only:
            #only ob used here
            return self._old_ob.copy(),_,_,{}
        else:
            return self._old_ob.copy(), 0.0, False, {}

    def _get_observation(self):
        q1, q2, dq1,dq2 = self._env.env.state
        return np.array([q1, q2, dq1,dq2])

    def _build_env(self):
        import gym
        self._current_version = gym.__version__
        _env_name = {
            'gym_robotarm': 'BaselineRobotArm-v0'
        }

        # make the environments
        self._env = gym.make(_env_name[self._env_name])
        self._env_info = env_register.get_env_info(self._env_name)

    def _set_groundtruth_api(self):
        """ @brief:
                In this function, we could provide the ground-truth dynamics
                and rewards APIs for the agent to call.
                For the new environments, if we don't set their ground-truth
                apis, then we cannot test the algorithm using ground-truth
                dynamics or reward
        """
        self._set_reward_api()
        self._set_dynamics_api()

    def _set_dynamics_api(self):

        def set_state(data_dict):
            # recover angles and velocities
            q1, q2, dq1,dq2 = data_dict['start_state'][0],data_dict['start_state'][1],data_dict['start_state'][2],data_dict['start_state'][3]
            state = np.asarray([q1, q2, dq1,dq2])
            # reset the state
            self._env.env.state = state
        self.set_state = set_state

        def fdynamics(data_dict):
            self.set_state(data_dict)
            action = data_dict['action']
            return self.step(action)[0]
        self.fdynamics = fdynamics

    def _set_reward_api(self):

        # step 1, set the zero-order reward function
        assert self._env_name in self.SWINGUP

        def reward(data_dict):
            state,action= data_dict['start_state'],data_dict['action']*self._env.env.max_torque
            reward = self._env.env.reward_func(state,action)
            return reward

        self.reward = reward

        def reward_derivative(data_dict, target):
            # return derivative_data
            print("...using derivative reward...")
            return None
        self.reward_derivative = reward_derivative


if __name__ == '__main__':

    test_env_name = ['gym_robotarm']
    for env_name in test_env_name:
        test_env = env(env_name, 1234, {})
        api_env = env(env_name, 1234, {})
        api_env.reset()
        ob,_,_,_ = test_env.reset()
        for _ in range(100):
            action = np.random.uniform(-1, 1, test_env._env.action_space.shape)
            new_ob, reward, _, _ = test_env.step(action)

            # test the reward api
            reward_from_api = \
                api_env.reward({'start_state': ob, 'action': action})
            reward_error = np.sum(np.abs(reward_from_api - reward))

            # test the dynamics api
            newob_from_api = \
                api_env.fdynamics({'start_state': ob, 'action': action})
            ob_error = np.sum(np.abs(newob_from_api - new_ob))

            ob = new_ob

            print('reward error: {}, dynamics error: {}'.format(
                reward_error, ob_error)
            )
            api_env._env.render()
            import time
            time.sleep(0.1)
