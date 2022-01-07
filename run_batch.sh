# !/bin/sh

n_proc=6
n_runs=2

# mpirun -n $n_proc python run_xp.py --n=$n_runs --n_pop=20  --evals=20000 --noise 0 --problem=leading_ones
# mpirun -n $n_proc python run_xp.py --n=$n_runs --n_pop=20  --evals=20000 --noise 0.25 --problem=leading_ones
# mpirun -n $n_proc python run_xp.py --n=$n_runs --n_pop=20  --evals=20000 --noise 0.5 --problem=leading_ones
# mpirun -n $n_proc python run_xp.py --n=$n_runs --n_pop=20  --evals=20000 --noise 0.75 --problem=leading_ones
# mpirun -n $n_proc python run_xp.py --n=$n_runs --n_pop=20  --evals=20000 --noise 1 --problem=leading_ones
# mpirun -n $n_proc python run_xp.py --n=$n_runs --n_pop=20  --evals=20000 --noise 2 --problem=leading_ones

# mpirun -n $n_proc python run_xp.py --n=$n_runs --n_pop=20  --evals=20000 --noise 0 --problem=float_all_ones
# mpirun -n $n_proc python run_xp.py --n=$n_runs --n_pop=20  --evals=20000 --noise 0.25 --problem=float_all_ones
# mpirun -n $n_proc python run_xp.py --n=$n_runs --n_pop=20  --evals=20000 --noise 0.5 --problem=float_all_ones
# mpirun -n $n_proc python run_xp.py --n=$n_runs --n_pop=20  --evals=20000 --noise 0.75 --problem=float_all_ones
# mpirun -n $n_proc python run_xp.py --n=$n_runs --n_pop=20  --evals=20000 --noise 1 --problem=float_all_ones
# mpirun -n $n_proc python run_xp.py --n=$n_runs --n_pop=20  --evals=20000 --noise 2 --problem=float_all_ones

mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=2000 --noise 0 --problem=cartpole --noise_type=action
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=2000 --noise 0.25 --problem=cartpole
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=2000 --noise 0.5 --problem=cartpole
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=2000 --noise 0.75 --problem=cartpole
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=2000 --noise 1 --problem=cartpole
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=2000 --noise 2 --problem=cartpole