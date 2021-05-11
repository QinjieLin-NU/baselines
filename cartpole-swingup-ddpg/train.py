import os

import math
import matplotlib.pyplot as plt
import torch

from network import DDPGAgent
from utils import *
import pickle
import config

def train(batch_size=128, critic_lr=1e-3, actor_lr=1e-4, max_episodes=1000, max_steps=500, gamma=0.99, tau=1e-3,
          buffer_maxlen=100000):
    # env = make("CartPoleSwingUpContinuous")
    env = make(config.env_name)

    max_episodes = max_episodes
    max_steps = max_steps
    batch_size = batch_size

    gamma = gamma
    tau = tau
    buffer_maxlen = buffer_maxlen
    critic_lr = critic_lr
    actor_lr = actor_lr

    agent = DDPGAgent(env, gamma, tau, buffer_maxlen, critic_lr, actor_lr, True, max_episodes * max_steps)
    evaluate_rewards,episode_rewards = mini_batch_train(env, agent, max_episodes, max_steps, batch_size)

    plt.figure()
    plt.plot(episode_rewards)
    plt.plot(evaluate_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")

    curr_dir = os.path.abspath(os.getcwd())
    if not os.path.isdir("models"):
        os.mkdir("models")
    torch.save(agent, curr_dir + "/models/%s_ddpg_pdp.pkl"%(config.env_name))

    plt.show()

    pickle.dump(episode_rewards, open("results/%s_ddpg_pdp_episode_reward.p"%config.env_name, "wb" )) 
    pickle.dump(evaluate_rewards, open("results/%s_ddpg_pdp_evaluate_reward.p"%config.env_name, "wb" )) 


def evaluate():
    # simulation of the agent solving the cartpole swing-up problem
    env = make(config.env_name)
    # uncomment for recording a video of simulation
    # env = Monitor(env, './video', force=True)

    curr_dir = os.path.abspath(os.getcwd())
    agent = torch.load(curr_dir + "/models/%s_ddpg_pdp.pkl"%(config.env_name))
    agent.train = False

    state = env.reset()
    r = 0
    theta = []
    actions = []
    for i in range(500):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        actions.append(action)
        # env.render()
        theta.append(math.degrees(next_state[2]))
        r += reward
        state = next_state
    print(state)

    env.close()

    # plot the angle and action curve
    curr_dir = os.path.abspath(os.getcwd())
    if not os.path.isdir("results"):
        os.mkdir("results")

    plt.figure()
    plt.plot(theta)
    plt.title('Angle')
    plt.ylabel('Angle in degrees')
    plt.xlabel('Time step t')
    plt.savefig(curr_dir + "/results/plot_angle.png")

    plt.figure()
    plt.plot(actions)
    plt.title('Action')
    plt.ylabel('Action in Newton')
    plt.xlabel('Time step t')
    plt.savefig(curr_dir + "/results/plot_action.png")


if __name__ == '__main__':
    # evaluate()
    # train(max_episodes=1200)
    train(max_episodes=1200)
