# !/bin/sh

n_proc=16
n_runs=3

# Binary
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 0 --problem=leading_ones
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 0.25 --problem=leading_ones
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 0.5 --problem=leading_ones
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 0.75 --problem=leading_ones
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 1 --problem=leading_ones
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 2 --problem=leading_ones

# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 0 --problem=all_ones
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 0.25 --problem=all_ones
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 0.5 --problem=all_ones
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 0.75 --problem=all_ones
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 1 --problem=all_ones
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 2 --problem=all_ones

# # Continuous
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 0 --problem=float_all_ones
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 0.25 --problem=float_all_ones
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 0.5 --problem=float_all_ones
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 0.75 --problem=float_all_ones
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 1 --problem=float_all_ones
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 2 --problem=float_all_ones

# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=2000 --noise 0 --problem=cartpole --noise_type=action
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=2000 --noise 0.25 --problem=cartpole --noise_type=action
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=2000 --noise 0.5 --problem=cartpole --noise_type=action 
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=2000 --noise 0.75 --problem=cartpole --noise_type=action
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=2000 --noise 1 --problem=cartpole --noise_type=action

# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=2000 --problem=cartpole --noise_type=seed

# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 0 --problem=min-breakout --noise_type=action --n_pop=16
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=20000 --noise 0 --problem=min-si --noise_type=action --n_pop=16

# Procgen
mpirun --use-hwthread-cpus -n $n_proc python run_xp.py --n=$n_runs --evals=100 --noise 0 --problem=bigfish --noise_type=seed --n_pop=32 --n_elites=6 --max_eval=48 --mut_size=0.01

# Performance comparison
# mpirun -n 1 python run_xp.py --n=1 --evals=2000 --noise 0 --problem=cartpole --noise_type=action --algos rs --no_plot 
# mpirun -n 6 python run_xp.py --n=1 --evals=2000 --noise 0 --problem=cartpole --noise_type=action --algos rs --no_plot
