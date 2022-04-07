# Policy-Dynamics Value Functions (PD-VF) 

This is source code for the paper 

[PAnDR: Fast Adaptation to New Environments from Offline Experiences via Decoupling Policy and Environment Representations]



## Envvironment install 
```
cd myant 
pip install -e .  

cd myswimmer 
pip install -e .  

cd myspaceship 
pip install -e .  
```

## (1) Reinforcement Learning Phase 

Train PPO policies on each environments.

Each of the commands below need to be run 
for seed in [0,...,4] and for default-ind in [0,...,19].

### Spaceship
```
python ppo/ppo_main.py \
--env-name spaceship-v0 --default-ind 0 --seed 0 
```

### Swimmer
```
python ppo/ppo_main.py \
--env-name myswimmer-v0 --default-ind 0 --seed 0 
```

### Ant-wind
```
python ppo/ppo_main.py \
--env-name myant-v0 --default-ind 0 --seed 0 
```

## (2) PAnDR Training Phase

## Dynamics Embedding

### Ant-wind
```
# python main_train.py --env-name myant-v0 --op-lr 0.01 --num-t-policy-embed 50 --num-t-env-embed 1 --gd-iter 50 --norm-reward --min-reward -200 --max-reward 1000 --club-lambda 1000 --mi-lambda 1
```



