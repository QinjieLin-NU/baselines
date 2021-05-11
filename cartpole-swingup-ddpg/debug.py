# from environment import CartPoleSwingUpContinuousEnv, RobotArm
# import numpy as np

# env = CartPoleSwingUpContinuousEnv()
# env.reset()
# env.state = [0.0,0.0,0.0,0.0]
# obs, reward, done, _ = env.step([10.0])
# print(obs,reward)
# obs, reward, done, _ = env.step([10.0])
# print(obs,reward)


# env=RobotArm()
# env.reset()
# state, reward ,_,_ = env.step(np.array([1.,2.]))
# print(state,reward)


import gym
import numpy as np
env = gym.make("BaselineSwingUp-v0")
s0 = env.reset()
a0 = np.array([0.1])
s1,_r,_,_ = env.step(a0)
print(s1)
s2,_r,_,_ = env.step(a0)
print(s2)

s1_hat = env.np_dynamics(s0,a0)
print(s1_hat)
s2_hat = env.np_dynamics(s1_hat,a0)
print(s2_hat)
