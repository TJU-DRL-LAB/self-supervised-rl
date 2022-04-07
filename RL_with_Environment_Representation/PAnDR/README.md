# ICLR 2022 GPL Workshop-Policy Adaptation via Decoupled Policy and Environment Representations (PAnDR)

This is the official implementation of 
our work [PAnDR: Fast Adaptation to New Environments from Offline Experiences via Decoupling Policy and Environment Representations](https://arxiv.org/abs/2204.02877)
presented at ICLR 2022 Workshop on Generalizable Policy Leanring (GPL).

## Introduction



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


We refer the user to our paper for complete details of hyperparameter settings and design choices.

## TO-DO
- [ ] Tidy up redundant codes

## Citation
If this repository has helped your research, please cite the following:
```bibtex
@inproceedings{sang2022pandr,
  title     = {PAnDR: Fast Adaptation to New Environments from Offline Experiences via Decoupling Policy and Environment Representations},
  author    = {Tong Sang, Hongyao Tang, Yi Ma, Jianye Hao, Yan Zheng, Zhaopeng Meng, Boyan Li, Zhen Wang},
  booktitle = {International Conference on Learning Representations Workshop on Generalizable Policy Learning},
  year      = {2022},
  url       = {https://arxiv.org/abs/2204.02877}
}
```

