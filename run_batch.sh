# !/bin/sh

n_proc=6
n_runs=3
wandb="ucea"

# Binary
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 0 --wandb=$wandb --problem=leading_ones 
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 0.25 --wandb=$wandb --problem=leading_ones 
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 0.5 --wandb=$wandb --problem=leading_ones 
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 0.75 --wandb=$wandb --problem=leading_ones 
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 1 --wandb=$wandb --problem=leading_ones 
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 2 --wandb=$wandb --problem=leading_ones 

# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 0 --wandb=$wandb --problem=all_ones
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 0.25 --wandb=$wandb --problem=all_ones
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 0.5 --wandb=$wandb --problem=all_ones
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 0.75 --wandb=$wandb --problem=all_ones
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 1 --wandb=$wandb --problem=all_ones
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 2 --wandb=$wandb --problem=all_ones

# # Continuous
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 0 --wandb=$wandb --problem=float_all_ones
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 0.25 --wandb=$wandb --problem=float_all_ones
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 0.5 --wandb=$wandb --problem=float_all_ones
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 0.75 --wandb=$wandb --problem=float_all_ones
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 1 --wandb=$wandb --problem=float_all_ones
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 2 --wandb=$wandb --problem=float_all_ones

# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=2000 --noise 0 --wandb=$wandb --problem=cartpole --noise_type=action --train_seeds=10000000 --val_seeds=10000000
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=2000 --noise 0.25 --wandb=$wandb --problem=cartpole --noise_type=action
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=2000 --noise 0.5 --wandb=$wandb --problem=cartpole --noise_type=action 
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=2000 --noise 0.75 --wandb=$wandb --problem=cartpole --noise_type=action
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=2000 --noise 1 --wandb=$wandb --problem=cartpole --noise_type=action --train_seeds=10000000 --val_seeds=10000000

mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=2000 --problem=cartpole --noise_type=seed --train_seeds=10000000 --val_seeds=10000000

# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 0 --wandb=$wandb --problem=min-breakout --noise_type=action --n_pop=16
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 0 --wandb=$wandb --problem=min-si --noise_type=action --n_pop=16

# Procgen
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=100 --noise 0 --wandb=$wandb --problem=bigfish --noise_type=seed --n_pop=32 --n_elites=6 --max_eval=48 --mut_size=0.01

# Performance comparison
# mpirun -n 1 python run_xp.py --n=1 --evals=2000 --noise 0 --wandb=$wandb --problem=cartpole --noise_type=action --algos rs --no_plot 
# mpirun -n 6 python run_xp.py --n=1 --evals=2000 --noise 0 --wandb=$wandb --problem=cartpole --noise_type=action --algos rs --no_plot
