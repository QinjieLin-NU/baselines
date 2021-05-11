import torch 

env_name = "CartPoleSwingUpContinuous" # CartPoleSwingUpContinuous    RobotArm Pendulum-v0
actor_multipler = 10.0 #10.0 for cartpole
step_reward = 0.0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

