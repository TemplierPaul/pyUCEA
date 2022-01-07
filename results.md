# Results

## Noisy sum

### Uniform noise

<!-- |%Noise|Fitness / Gen |Fitness / Eval| Cost / Gen |
|---|---|---|---|
|0%|![](plots/Uniform/Gen_All_Ones_0.png)|![](plots/Uniform/Eval_All_Ones_0.png)|![](plots/Uniform/Cost_All_Ones_0.png)
|25%|![](plots/Uniform/Gen_All_Ones_25.png)|![](plots/Uniform/Eval_All_Ones_25.png)|![](plots/Uniform/Cost_All_Ones_25.png)
|50%|![](plots/Uniform/Gen_All_Ones_50.png)|![](plots/Uniform/Eval_All_Ones_50.png)|![](plots/Uniform/Cost_All_Ones_50.png)
|75%|![](plots/Uniform/Gen_All_Ones_75.png)|![](plots/Uniform/Eval_All_Ones_75.png)|![](plots/Uniform/Cost_All_Ones_75.png)
|100%|![](plots/Uniform/Gen_All_Ones_100.png)|![](plots/Uniform/Eval_All_Ones_100.png)|![](plots/Uniform/Cost_All_Ones_100.png)
|200%|![](plots/Uniform/Gen_All_Ones_200.png)|![](plots/Uniform/Eval_All_Ones_200.png)|![](plots/Uniform/Cost_All_Ones_200.png) -->

|%Noise|Fitness / Eval
|---|---|
|0%|![](plots/Uniform/Eval_F_All_Ones_0.png)
|25%|![](plots/Uniform/Eval_F_All_Ones_25.png)
|50%|![](plots/Uniform/Eval_F_All_Ones_50.png)
|75%|![](plots/Uniform/Eval_F_All_Ones_75.png)
|100%|![](plots/Uniform/Eval_F_All_Ones_100.png)
|200%|![](plots/Uniform/Eval_F_All_Ones_200.png)


## Float All Ones

- Vector or 10 floating point values
- Goal: find a vector of ones
- Fitness: $$10 - MSE(genome, [1, ..., 1])$$

### Uniform noise

|%Noise|Fitness / Eval
|---|---|
|0%|![](plots/Uniform/Eval_F_Float_All_Ones_0.png)
|25%|![](plots/Uniform/Eval_F_Float_All_Ones_25.png)
|50%|![](plots/Uniform/Eval_F_Float_All_Ones_50.png)
|75%|![](plots/Uniform/Eval_F_Float_All_Ones_75.png)
|100%|![](plots/Uniform/Eval_F_Float_All_Ones_100.png)
|200%|![](plots/Uniform/Eval_F_Float_All_Ones_200.png)

## Leading Ones
### Uniform noise

|%Noise|Fitness / Eval
|---|---|
|0%|![](plots/Uniform/Eval_F_Leading_Ones_0.png)
|25%|![](plots/Uniform/Eval_F_Leading_Ones_25.png)
|50%|![](plots/Uniform/Eval_F_Leading_Ones_50.png)
|75%|![](plots/Uniform/Eval_F_Leading_Ones_75.png)
|100%|![](plots/Uniform/Eval_F_Leading_Ones_100.png)
|200%|![](plots/Uniform/Eval_F_Leading_Ones_200.png)


## CartPole 
- Gym CartPole-v1
- 10 neurons in the hidden layer
- Max 200 steps


### Noise on actions
|%Noise|Fitness / Eval
|---|---|
|0%|![](plots/Uniform/Eval_A_CartPole-v1_0.png)
|25%|![](plots/Uniform/Eval_A_CartPole-v1_25.png)
|50%|![](plots/Uniform/Eval_A_CartPole-v1_50.png)
|75%|![](plots/Uniform/Eval_A_CartPole-v1_75.png)
|100%|![](plots/Uniform/Eval_A_CartPole-v1_100.png)

## Noise on seed
(Ignore the noise % in the title)
![](plots/Uniform/Eval_S_CartPole-v1_0.png)


## Minatar 

### Noise on actions
|%Noise|Breakout | Space Invaders
|---|---|---|
|0%|![](plots/Uniform/Eval_A_min-breakout_0.png)|![](plots/Uniform/Eval_A_min-space_invaders_0.png)
|25%|![](plots/Uniform/Eval_A_min-breakout_25.png)|![](plots/Uniform/Eval_A_min-space_invaders_25.png)
|50%|![](plots/Uniform/Eval_A_min-breakout_50.png)|![](plots/Uniform/Eval_A_min-space_invaders_50.png)
|75%|![](plots/Uniform/Eval_A_min-breakout_75.png)|![](plots/Uniform/Eval_A_min-space_invaders_75.png)
|100%|![](plots/Uniform/Eval_A_min-breakout_100.png)|![](plots/Uniform/Eval_A_min-space_invaders_100.png)

## Noise on seed
(Ignore the noise % in the title)
![](plots/Uniform/Eval_S_min-breakout_0.png)|![](plots/Uniform/Eval_S_min-space_invaders_0.png)