import numpy as np
# from tensorflow.python.summary.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_tensorflow_log(path):

    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': 100,
        'histograms': 1
    }

    event_acc = EventAccumulator(path, tf_size_guidance)
    event_acc.Reload()

    # Show all tags in the log file
    print(event_acc.Tags())


    train_rewards =   event_acc.Scalars('avg_reward')
    values = []
    for i in train_rewards:
        values.append(i.value)
    plt.plot(values)
    plt.show()


if __name__ == '__main__':
    log_file = "/root/mbbl/log/mbrl-cemgym_pendulumrs_gym_pendulum.log/mbrl-cemgym_pendulumrs_gym_pendulum.log"
    plot_tensorflow_log(log_file)