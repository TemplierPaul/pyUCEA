# Results

## Noisy sum
- Vector or 10 floating point values
- Goal: find a vector of ones
- Fitness: $$10 - MSE(genome, [1, ..., 1])$$
### Uniform noise

|%Noise|Fitness / Gen |Fitness / Eval| Cost / Gen |
|---|---|---|---|
<!-- |0%|![](plots/Uniform/Gen_All_Ones_0.png)|![](plots/Uniform/Eval_All_Ones_0.png)|![](plots/Uniform/Cost_All_Ones_0.png)
|25%|![](plots/Uniform/Gen_All_Ones_25.png)|![](plots/Uniform/Eval_All_Ones_25.png)|![](plots/Uniform/Cost_All_Ones_25.png)
|50%|![](plots/Uniform/Gen_All_Ones_50.png)|![](plots/Uniform/Eval_All_Ones_50.png)|![](plots/Uniform/Cost_All_Ones_50.png)
|75%|![](plots/Uniform/Gen_All_Ones_75.png)|![](plots/Uniform/Eval_All_Ones_75.png)|![](plots/Uniform/Cost_All_Ones_75.png)
|100%|![](plots/Uniform/Gen_All_Ones_100.png)|![](plots/Uniform/Eval_All_Ones_100.png)|![](plots/Uniform/Cost_All_Ones_100.png)
|200%|![](plots/Uniform/Gen_All_Ones_200.png)|![](plots/Uniform/Eval_All_Ones_200.png)|![](plots/Uniform/Cost_All_Ones_200.png) -->

|0%|![](plots/Uniform/Eval_All_Ones_0.png)
|25%|![](plots/Uniform/Eval_All_Ones_25.png)
|50%|![](plots/Uniform/Eval_All_Ones_50.png)
|75%|![](plots/Uniform/Eval_All_Ones_75.png)
|100%|![](plots/Uniform/Eval_All_Ones_100.png)
|200%|![](plots/Uniform/Eval_All_Ones_200.png)

### Normal noise
|%Noise|Fitness / Gen |Fitness / Eval| Cost / Gen |
|---|---|---|---|
|0%|![](plots/Normal/Gen_All_Ones_0.png)|![](plots/Normal/Eval_All_Ones_0.png)|![](plots/Normal/Cost_All_Ones_0.png)
|25%|![](plots/Normal/Gen_All_Ones_25.png)|![](plots/Normal/Eval_All_Ones_25.png)|![](plots/Normal/Cost_All_Ones_25.png)
|50%|![](plots/Normal/Gen_All_Ones_50.png)|![](plots/Normal/Eval_All_Ones_50.png)|![](plots/Normal/Cost_All_Ones_50.png)
|75%|![](plots/Normal/Gen_All_Ones_75.png)|![](plots/Normal/Eval_All_Ones_75.png)|![](plots/Normal/Cost_All_Ones_75.png)
|100%|![](plots/Normal/Gen_All_Ones_100.png)|![](plots/Normal/Eval_All_Ones_100.png)|![](plots/Normal/Cost_All_Ones_100.png)



## CartPole 
- Gym CartPole-v1
- 10 neurons in the hidden layer
- Max 200 steps


### Uniform noise

|%Noise|Fitness / Gen |Fitness / Eval| Cost / Gen |
|---|---|---|---|
|0%|![](plots/Uniform/Gen_CartPole-v1_0.png)|![](plots/Uniform/Eval_CartPole-v1_0.png)|![](plots/Uniform/Cost_CartPole-v1_0.png)
|25%|![](plots/Uniform/Gen_CartPole-v1_25.png)|![](plots/Uniform/Eval_CartPole-v1_25.png)|![](plots/Uniform/Cost_CartPole-v1_25.png)
|50%|![](plots/Uniform/Gen_CartPole-v1_50.png)|![](plots/Uniform/Eval_CartPole-v1_50.png)|![](plots/Uniform/Cost_CartPole-v1_50.png)
|75%|![](plots/Uniform/Gen_CartPole-v1_75.png)|![](plots/Uniform/Eval_CartPole-v1_75.png)|![](plots/Uniform/Cost_CartPole-v1_75.png)
|100%|![](plots/Uniform/Gen_CartPole-v1_100.png)|![](plots/Uniform/Eval_CartPole-v1_100.png)|![](plots/Uniform/Cost_CartPole-v1_100.png)

### Normal noise

|%Noise|Fitness / Gen |Fitness / Eval|
|---|---|---|
|0%|![](plots/Normal/Gen_CartPole-v1_0.png)|![](plots/Normal/Eval_CartPole-v1_0.png)
|25%|![](plots/Normal/Gen_CartPole-v1_25.png)|![](plots/Normal/Eval_CartPole-v1_25.png)
|50%|![](plots/Normal/Gen_CartPole-v1_50.png)|![](plots/Normal/Eval_CartPole-v1_50.png)
|75%|![](plots/Normal/Gen_CartPole-v1_75.png)|![](plots/Normal/Eval_CartPole-v1_75.png)
|100%|![](plots/Normal/Gen_CartPole-v1_100.png)|![](plots/Normal/Eval_CartPole-v1_100.png)
