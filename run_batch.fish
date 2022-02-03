#!usr/bin/fish

set n_proc 4
set n_runs 1
set job 4600
set wandb "lucie-robot"

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

# Procgen
# mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=100 --noise 0 --wandb=$wandb --problem=bigfish --noise_type=seed --n_pop=32 --n_elites=6 --max_eval=48 --mut_size=0.01

for problem in cartpole acrobot
	for noise in 0.75 0.5 0.25 0.0
		echo "$job $problem $noise"
		for i in (seq 1 4)
			mpirun -n $n_proc python run_xp.py --n=$n_runs --evals=50000 --noise $noise --problem $problem --noise_type sticky --n_pop 100 --n_elites 10 --max_eval 1000 --tournament_fitness median --mut_size 0.05 --val_size 20 --val_freq 5 --epsilon 3 --no_plot --scaling_type scale --train_seeds 0 --val_seeds 0 --n_resampling 10 --wandb $wandb --job $job --thread $i --entity sureli &
		end
		wait
		set job (math $job+1)
	end
	set job (math $job+50)
end
