# FrozenLake-v1 Deep Q-Learning

This repository branch provides pre-trained agents (models) as well as easy-to-use customised code for training and inferring an agent to play the [FrozenLake-v1](https://gym.openai.com/envs/FrozenLake-v0/), a Toy Text game built by [OpenAI](https://openai.com/) under [OpenAI Gym](https://gym.openai.com/) that officially is a toolkit for developing and comparing reinforcement learning algorithms.

## I. DIY Guide

### System Requirements
* Linux or Mac OS
* Python 3.6 or above

### Installation and Setup
* Install dependencies using [requirements.txt](requirements.txt)
```
    pip install -r requirements.txt
```

### Training Flow
* #### Pre-Training Steps
  * Edit the following params from [FrozenLakeDQN.yml](config/FrozenLakeDQN.yml) :
    * ENVIRONMENT:
      * ```env_name```: 'FrozenLake-v1' 
      
      * ```custom_map_flag```: False 
      
      * ```custom_map```: [
          'SFFHF',
          'HFHFF',
          'HFFFH',
          'HHHFH',
          'HFFFG'
         ]  # Define custom_map of n x n dimensions only if custom_map_flag is set True
         
      * ```render_flag```: False
      
      * ```rewards```:
      
        * ```positive_step_reward```: 0.25 &emsp; # Reward for ending up on FROZEN surface 
        
        * ```negative_step_reward```: -1.0 &emsp; # Reward for ending up in HOLE 
        
        * ```goal_step_reward```: 1.0 &emsp; &emsp; # Reward for ending up to GOAL

    * AGENT:
      * ```total_episodes```: 50000 
      
      * ```gamma```: 0.97 
      
      * ```epsilon```: 1.0 
      
      * ```min_epsilon```: 0.01 
      
      * ```training_data_deque_max_len```: 1000 &emsp; &emsp; # Max length for deque array that stores training data 
      
      * ```train_start_threshold```: 500 &emsp; &emsp; &emsp; # Starts training model after length of training_data_deque_max_len exceeds given value 
      
      * ```load_model_flag```: False 
      
      * ```model_path```: "" &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; # Define model_path if load_model_flag is True

    * NN_MODEL:
      * ```neuron_units```: [32, 64]  &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; # [hidden layer1, hidden layer2]
      * ```activations```: ['swish', 'swish', 'linear']  &emsp; # [layer1, layer2, output_layer] Refer https://keras.io/api/layers/activations/
      
      * ```kernel_initializers```: ['he_uniform', 'he_uniform', 'he_uniform']  &ensp; # Refer https://keras.io/api/layers/initializers/
      
      * ```loss```: 'mse'  &emsp; &emsp; &emsp; &emsp; &emsp; &ensp; &ensp; &emsp; &emsp; &emsp; &emsp; # Refer https://keras.io/api/losses/
      
      * ```learning_rate```: 0.01 
      
      * ```batch_size```: 32 
      
      * ```model_checkpoint```: 50  &emsp; &emsp; &emsp; &emsp; &emsp; &ensp; # Saves model after every given episodes

* #### Initiate Training
  * Run [FrozenLakeDQN.py](FrozenLakeDQN.py) using following command:
  ```
        python3 FrozenLakeDQN.py
  ```

* #### Post-Training Data Visualization
  * Training plots, training logs, and trained models can be found here:
    * [Training Logs](Training/TrainingLogs/FrozenLakeDQN)
    * [Training Plots](Training/TrainingPlots/FrozenLakeDQN)
    * [Trained Models](Training/TrainedModels/FrozenLakeDQN)


### Inference Flow
* #### Initiate Training
  * Run [FrozenLakeDQNInference.py](FrozenLakeDQNInference.py) using following command:
  ```
    python3 FrozenLakeDQNInference.py <path_to_model>
  ```

* #### Post-Inference Data Visualization
  * Inference plots and logs can be found here:
    * [Training Logs](Inference/InferenceLogs/FrozenLakeDQN)
    * [Training Plots](Inference/InferencePlots/FrozenLakeDQN)

**Note**: Name format for training and inference data files is : ```env_name``` + ```_``` + instantaneous_timestamp. Eg:- FrozenLake-v1_15-Sep-2021_21-30-35

## II. Pre-Trained Model Inference

### Model-1
* [Model-1](Training/TrainedModels/FrozenLakeDQN/FrozenLake-v1_14-Sep-2021_20-01-42.h5) is trained considering the parameters from [FrozenLakeDQN.yml](config/FrozenLakeDQN.yml) from FrozenLakeDQN-M1 tag.

### Model-1 Inference
#### Inference-1 &emsp; &emsp; Prob(Successful Chase) = 0.62 &emsp; &emsp; Inference Logs = [FrozenLake-v1_15-Sep-2021_21-30-35.log](Inference/InferenceLogs/FrozenLakeDQN/FrozenLake-v1_15-Sep-2021_21-30-35.log)
<img height="400" src="Inference/InferencePlots/FrozenLakeDQN/FrozenLake-v1_15-Sep-2021_21-30-35.png" width="400" alt="Inference-1" title="Inference-1"/>

#### Inference-2 &emsp; &emsp; Prob(Successful Chase) = 0.53 &emsp; &emsp; Inference Logs = [FrozenLake-v1_15-Sep-2021_22-19-45.log](Inference/InferenceLogs/FrozenLakeDQN/FrozenLake-v1_15-Sep-2021_22-19-45.log)
<img height="400" src="Inference/InferencePlots/FrozenLakeDQN/FrozenLake-v1_15-Sep-2021_22-19-45.png" width="400" alt="Inference-2" title="Inference-2"/>

#### Inference-3 &emsp; &emsp; Prob(Successful Chase) = 0.56 &emsp; &emsp; Inference Logs = [FrozenLake-v1_15-Sep-2021_22-29-48.log](Inference/InferenceLogs/FrozenLakeDQN/FrozenLake-v1_15-Sep-2021_22-29-48.log)
<img height="400" src="Inference/InferencePlots/FrozenLakeDQN/FrozenLake-v1_15-Sep-2021_22-29-48.png" width="400" alt="Inference-3" title="Inference-3"/>

#### Inference-4 &emsp; &emsp; Prob(Successful Chase) = 0.57 &emsp; &emsp; Inference Logs = [FrozenLake-v1_15-Sep-2021_22-40-40.log](Inference/InferenceLogs/FrozenLakeDQN/FrozenLake-v1_15-Sep-2021_22-40-40.log)
<img height="400" src="Inference/InferencePlots/FrozenLakeDQN/FrozenLake-v1_15-Sep-2021_22-40-40.png" width="400" alt="Inference-4" title="Inference-4"/>

#### Inference-5 &emsp; &emsp; Prob(Successful Chase) = 0.56 &emsp; &emsp; Inference Logs = [FrozenLake-v1_15-Sep-2021_22-50-30.log](Inference/InferenceLogs/FrozenLakeDQN/FrozenLake-v1_15-Sep-2021_22-50-30.log)
<img height="400" src="Inference/InferencePlots/FrozenLakeDQN/FrozenLake-v1_15-Sep-2021_22-50-30.png" width="400" alt="Inference-5" title="Inference-5"/>

#### Inference-6 &emsp; &emsp; Prob(Successful Chase) = 0.54 &emsp; &emsp; Inference Logs = [FrozenLake-v1_15-Sep-2021_23-07-21.log](Inference/InferenceLogs/FrozenLakeDQN/FrozenLake-v1_15-Sep-2021_23-07-21.log)
<img height="400" src="Inference/InferencePlots/FrozenLakeDQN/FrozenLake-v1_15-Sep-2021_23-07-21.png" width="400" alt="Inference-6" title="Inference-6"/>




## Change Logs
* Implementation of Q-Learning for FrozenLake-v1
* Implementation of Deep Q-Learning for FrozenLake-v1
* Added custom map support for FrozenLake-v1 environment
* Trained Model-1 considering configurations from [FrozenLakeDQN.yml](config/FrozenLakeDQN.yml) from FrozenLakeDQN-M1 tag


