mkdir /root/miniconda3/lib/python3.8/site-packages/gym/envs/baseline_env
cd /root/
git clone https://github.com/QinjieLin-NU/baselines.git
cp /root/baselines/cartpole-swingup-ddpg/environment.py /root/miniconda3/lib/python3.8/site-packages/gym/envs/baseline_env/
echo  "
\n
register(
    id='BaselineSwingUp-v0',
    entry_point='gym.envs.baseline_env.environment:CartPoleSwingUpContinuousEnv',
    max_episode_steps=200,
    reward_threshold=25.0,
)
register(
    id='BaselineRobotArm-v0',
    entry_point='gym.envs.baseline_env.environment:RobotArm',
    max_episode_steps=200,
    reward_threshold=25.0,
)\n
" >> /root/miniconda3/lib/python3.8/site-packages/gym/envs/__init__.py 

