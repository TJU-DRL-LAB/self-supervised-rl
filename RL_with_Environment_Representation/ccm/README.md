# AAAI 2021-Towards Effective Context for Meta-Reinforcement Learning: an Approach based on Contrastive Learning (CCM)

This is the official implementation of 
our work [HyAR: Addressing Discrete-Continuous Action Reinforcement Learning via Hybrid Action Representation](https://openreview.net/forum?id=64trBbOhdGU)
accepted on ICLR 2022.

## Introduction

Discrete-continuous hybrid action space is a natural setting in many practical problems, such as robot control and game AI. However, most previous Reinforcement Learning (RL) works only demonstrate the success in controlling with either discrete or continuous action space, while seldom take into account the hybrid action space. 

One naive way to address hybrid action RL is to convert the hybrid action space into a unified homogeneous action space by discretization or continualization, so that conventional RL algorithms can be applied. 
However, this **ignores the underlying structure of hybrid action space** and also induces the scalability issue and additional approximation difficulties, thus leading to degenerated results. 

In this work, we propose **Hybrid Action Representation (HyAR)** to learn a **compact** and **decodable** latent representation space for the original hybrid action space:
- HyAR constructs the latent space and embeds the dependence between discrete action and continuous parameter via an embedding table and conditional Variantional Auto-Encoder (VAE).
- To further improve the effectiveness, the action representation is trained to be semantically smooth through unsupervised environmental dynamics prediction.
- Finally, the agent then learns its policy with conventional DRL algorithms in the learned representation space and interacts with the environment by decoding the hybrid action embeddings to the original action space. 

A conceptual illustration is shown below.

<div align=center><img align="center" src="./assets/HyAR_concept.png" alt="HyAR Conceptual Illustration" style="zoom:40%;" /></div>



## Related Work

This repo includes several reinforcement learning algorithms for hybrid action space MDPs:
1. HPPO[[Fan et al. 2018]](https://arxiv.org/abs/1903.01344v3)
2. MAHHQN[[Fu et al. 2018]](https://arxiv.org/abs/1903.04959)
3. P-DQN [[Xiong et al. 2018]](https://arxiv.org/abs/1810.06394)
4. PA-DDPG [[Hausknecht & Stone 2016]](https://arxiv.org/abs/1511.04143)


## Repo Content

### Folder Description
- configs: The configs to run different experiments
- rlkit：policies, samplers, networks, etc
- rand_param_envs: submodule rand_param_envs for a self-contained repo as in PEARL[[Rakelly et al., 2019]](https://arxiv.org/abs/1903.08254v1)

### Domains

Experiment scripts are provided to run our algorithm on the OpenAI gym, with the MuJoCo simulator, We further modify the original tasks to be Meta-RL tasks similar to PEARL.



## Installation

Here is an ancient installation guidance which needs step-by-step installation. A more automatic guidance with pip will be considered in the future.

To install locally, you will need to first install MuJoCo. For the task distributions in which the reward function varies (Cheetah, Ant, Humanoid), install MuJoCo200. Set LD_LIBRARY_PATH to point to the MuJoCo binaries (/$HOME/.mujoco/mujoco200/bin)

We recommend the user to install **anaconada** for convenient management of different python envs.

### Dependencies

- Python 3.6+ (tested with 3.6 and 3.7)
- torch 1.10+

other dependencies are same as PEARL

## Example Usage

CCM:
```bash
python launch_experiment.py ./configs/cheetah-sparse.json --gpu=0 --seed=0 --exp_id=ccm
```
above command will save log in (log/ccm/cheeta-sparse/0)

plot:
```bash
python plot_csv.py --id ccm --env_name cheetah-sparse --entry "AverageReturn_all_train_tasks_last" --add_tag _tag --seed 0 1 2
```

We refer the user to our paper for complete details of hyperparameter settings and design choices.

## TO-DO
- [ ] Tidy up redundant codes

## Citation
If this repository has helped your research, please cite the following:
```bibtex
@inproceedings{li2022hyar,
  title     = {Hy{AR}: Addressing Discrete-Continuous Action Reinforcement Learning via Hybrid Action Representation},
  author    = {Boyan Li and Hongyao Tang and YAN ZHENG and Jianye HAO and Pengyi Li and Zhen Wang and Zhaopeng Meng and LI Wang},
  booktitle = {International Conference on Learning Representations},
  year      = {2022},
  url       = {https://openreview.net/forum?id=64trBbOhdGU}
}
```
