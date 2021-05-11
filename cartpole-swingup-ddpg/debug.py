from environment import CartPoleSwingUpContinuousEnv, RobotArm
import numpy as np

# env = CartPoleSwingUpContinuousEnv()
# env.reset()
# env.state = [0.0,0.0,0.0,0.0]
# obs, reward, done, _ = env.step([10.0])
# print(obs,reward)
# obs, reward, done, _ = env.step([10.0])
# print(obs,reward)


env=RobotArm()
env.reset()
state, reward ,_,_ = env.step(np.array([1.,2.]))
print(state,reward)