for env_type in $1; do
    python main/pets_main.py --exp_id pets_${env_type}\
        --task $env_type \
        --num_planning_traj 500 --planning_depth $2 --random_timesteps 0 \
        --timesteps_per_batch 1 --num_workers 10 --max_timesteps 20000 \
        --gt_dynamics 1
done