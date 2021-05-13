import numpy as np
# from tensorflow.python.summary.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib as mpl
import matplotlib.pyplot as plt

episode_lenth={
    "pendulum": 200,
    "robotarm": 200,
    "cartpole":500
}

data_ts={
    "PPO": 2000, #batch_soze
    "MBMF": 1000, #batch_size
}

def get_data(path):
    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': 1000,
        'histograms': 1
    }

    event_acc = EventAccumulator(path)
    event_acc.Reload()
    print(event_acc.Tags())

    # Show all tags in the log file
    train_rewards =  event_acc.Scalars('avg_reward')
    values = []
    for i in train_rewards:
        if(i.value<=0):
            values.append(i.value)
    return values

def transfer_episode(ts_data,env_name,method_name):
    repeat_num = int( data_ts[method_name] / episode_lenth[env_name])
    print(method_name,env_name,repeat_num)
    _data = np.array(ts_data)
    transfered_data = np.repeat(_data,repeat_num)
    return list(transfered_data)

def get_dict(env_name,pets_log_file,mbmf_log_file,ppo_log_file):
    log_files = {
        "PETS":pets_log_file, #save per 200
        "MBMF":mbmf_log_file, #save per 1k, related to batch size
        "PPO":ppo_log_file   #save per2k, related to batch size
    }
    data_dict={
        "PETS":[],
        "MBMF":[],
        "PPO":[]
    }
    
    for _m in log_files:
        _data = get_data(log_files[_m])
        if(_m == "PPO"):
            _data = transfer_episode(_data,env_name,_m)
        elif(_m=="MBMF"):
            _data = transfer_episode(_data,env_name,_m)
        data_dict[_m] = _data
    return data_dict

def get_pendulum():
    pets_log_file = "log/mbrl-cemgym_pendulumpets_gym_pendulum.log/mbrl-cemgym_pendulumpets_gym_pendulum.log"
    mbmf_log_file = "log/mbmfrl-rsgym_pendulummbmf_gym_pendulum_ppo_seed_4123.log/mbmfrl-rsgym_pendulummbmf_gym_pendulum_ppo_seed_4123.log"
    ppo_log_file = "log/mfrl-mfgym_pendulumppo_gym_pendulum_batch_2000_seed_1234.log/mfrl-mfgym_pendulumppo_gym_pendulum_batch_2000_seed_1234.log"
    data_dict = get_dict("pendulum",pets_log_file,mbmf_log_file,ppo_log_file)
    return data_dict

def get_cartpole():
    pets_log_file = "log/mbrl-cemgym_swinguppets_gym_swingup.log/mbrl-cemgym_swinguppets_gym_swingup.log"
    mbmf_log_file = "log/mbmfrl-rsgym_swingupmbmf_gym_swingup_ppo_seed_2341.log/mbmfrl-rsgym_swingupmbmf_gym_swingup_ppo_seed_2341.log"
    ppo_log_file = "log/mfrl-mfgym_swingupppo_gym_swingup_batch_2000_seed_1234.log/mfrl-mfgym_swingupppo_gym_swingup_batch_2000_seed_1234.log"

    data_dict = get_dict("cartpole",pets_log_file,mbmf_log_file,ppo_log_file)
    return data_dict

def get_robotarm():
    pets_log_file = "log/mbrl-cemgym_robotarmpets_gym_robotarm.log/mbrl-cemgym_robotarmpets_gym_robotarm.log"
    mbmf_log_file = "log/mbmfrl-rsgym_robotarmmbmf_gym_robotarm_ppo_seed_1234.log/mbmfrl-rsgym_robotarmmbmf_gym_robotarm_ppo_seed_1234.log"
    ppo_log_file = "log/mfrl-mfgym_robotarmppo_gym_robotarm_batch_2000_seed_1234.log/mfrl-mfgym_robotarmppo_gym_robotarm_batch_2000_seed_1234.log"

    data_dict = get_dict("robotarm",pets_log_file,mbmf_log_file,ppo_log_file)
    return data_dict

if __name__ == '__main__':
    datas = get_pendulum()
    for _m in datas:
        plt.plot(datas[_m])
    plt.show()
    
    datas = get_cartpole()
    for _m in datas:
        plt.plot(datas[_m])
    plt.show()

    datas = get_robotarm()
    for _m in datas:
        plt.plot(datas[_m])
    plt.show()