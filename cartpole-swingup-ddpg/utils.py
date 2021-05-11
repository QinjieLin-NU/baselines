from environment import CartPoleSwingUpContinuousEnv,RobotArm
from gym import make as gym_make
import config

def make(env_name, *make_args, **make_kwargs):
    if env_name == "CartPoleSwingUpContinuous":
        return CartPoleSwingUpContinuousEnv()
    elif env_name == "RobotArm":
        return RobotArm()
    else:
        return gym_make(env_name, *make_args, **make_kwargs)


def mini_batch_train(env, agent, max_episodes, max_steps, batch_size):
    episode_rewards = []
    evaluate_rewards = []
    counter = 0

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state, (episode + 1) * (step + 1))
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward

            # update the agent if enough transitions are stored in replay buffer
            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)

            if done or step == max_steps - 1:
                episode_rewards.append(episode_reward)
                # Count number of consecutive games with cumulative rewards >-55 for early stopping
                if episode_reward > config.step_reward: #-55.0
                    counter += 1
                else:
                    counter = 0
                print("Episode " + str(episode) + " episodic reward: " + str(episode_reward))
                break

            state = next_state
        #evaluate current policy
        evaluate_rewards.append( evaluate(env,agent) )
        print("Episode " + str(episode) + " evaluate reward: " + str(evaluate_rewards[-1]))
        # Early stopping, if cumulative rewards of 10 consecutive games were >-55
        if counter == 10:
            break

    return evaluate_rewards,episode_rewards

def evaluate(env,agent):
    agent.train = False

    state = env.reset()
    r = 0
    for i in range(500):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        r += reward
        state = next_state
    agent.train=True
    return r
