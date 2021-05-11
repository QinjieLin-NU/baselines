#make sure cartpole_swing is in the Volume, urdf path, ignore torch install in req.txt
pip install ray
pip install 'ray[rllib]'
pip install tensorflow
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install raylab
pip install seaborn
#add envs to gym
cp -r /root/qinjie-vol/cartpole_swingup /root/cartpole_swingup
cd /root/cartpole_swingup
pip install -r requirements.txt
cd /root/
cp -r /root/cartpole_swingup /root/miniconda3/lib/python3.8/site-packages/gym/envs/.
#add envs to gym
mkdir /root/miniconda3/lib/python3.8/site-packages/gym/envs/cartpole_swingup/ddpg_env
cd /root/
git clone https://github.com/immvp1995/cartpole-swingup-ddpg.git
cp /root/cartpole-swingup-ddpg/environment.py /root/miniconda3/lib/python3.8/site-packages/gym/envs/cartpole_swingup/ddpg_env
echo "export PYTHONPATH=$PYTHONPATH:/root/cartpole_swingup/" >> ~/.bashrc
git clone https://github.com/ray-project/ray.git
git clone https://github.com/angelolovatto/raylab.git
echo  "
\n
register(
    id='SwingUp-v0',
    entry_point='gym.envs.cartpole_swingup.environment:CartPoleSwingUpContinuousEnv',
    max_episode_steps=500,
    reward_threshold=25.0,
)
register(
    id='SwingUp-v1',
    entry_point='gym.envs.cartpole_swingup.ddpg_env.environment:CartPoleSwingUpContinuousEnv',
    max_episode_steps=500,
    reward_threshold=25.0,
)\n
" >> /root/miniconda3/lib/python3.8/site-packages/gym/envs/__init__.py 
