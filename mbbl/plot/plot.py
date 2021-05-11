from packaging import version

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb

experiment_id = "pets-gt-gym_cheetah"
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
df = experiment.get_scalars()