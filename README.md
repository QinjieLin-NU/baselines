# baselines

## install dependency in container

```
bash install_conda.sh
bash install_ray.sh
bash install_baseline.sh
```

## set up ddpg

```
change config.py
python train.py
```

## set up baselines (ppo,pets,mbmf)

```
bash scripts/exp_nips/ppo.sh gym_swingup
bash scripts/exp_nips/pets.sh gym_swingup 30
bash scripts/exp_nips/mbmf.sh gym_swingup
```

## check reward curve

```
tensorboard --logdir log/  --host 0.0.0.0
```
