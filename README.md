# Self-supervised RL

A unified opensource code implementation of algorithms for Self-supervised Reinforcement Leanring (SSRL) with Representations.

This repo contains representative research works of TJU-RL-Lab on the topic of Self-supervised Representation Learning for RL.

This repo will be constantly updated to include new researches made by TJU-RL-Lab. 
(The development of this repo is in progress at present.)




## Introduction
**[A brief intro to SSRL (?)]**


### Taxonomy of SSRL
This repo follow a systematic taxnomy of Self-supervised RL with Representations proposed by TJU-RL-Lab, which consists of:
- SSRL with State Representation
- SSRL with Action Representation
- SSRL with Policy Representation
- SSRL with Environment (and Task) Representation
- SSRL with Other Representation

For a tutorial of this taxnomy, we refer the reader to our [ZhiHu blog series](https://zhuanlan.zhihu.com/p/413321572).

### Ecology of SSRL

<div align=center>![Ecology of SSRL](http://rl.beiyang.ren/tju_rl/self-supervised-rl/raw/master/assets/Ecology_of_SSRL.png)</div>

### The Unified Implementation Design of SSRL Algorithm

<div align=center>![Ecology of SSRL](http://rl.beiyang.ren/tju_rl/self-supervised-rl/raw/master/assets/alg_framework.png)</div>


## Installation

The algorithms in this repo are all implemented **python 3.5** (and versions above). 
**Tensorflow 1.x** and **PyTorch** are the main DL code frameworks we adopt in this repo with different choices in different algorithms.

First of all, we recommend the user to install **anaconada** and or **venv** for convenient management of different python envs.

In this repo, the following RL environments may be needed:
- [OpenAI Gym](https://github.com/openai/gym) (e.g., MuJoCo, Robotics)
- [MinAtar](https://github.com/kenjyoung/MinAtar)
- xxxx
- And some environments designed by ourselves.

Note that each algorithm may use only one or several environments in the ones listed above. Please refer to the page of specific algorithm for concrete requirements.

To clone this repo:

```
git clone http://rl.beiyang.ren/tju_rl/self-supervised-rl.git
```

Note that this repo is a collection of multiple research branches (according to the taxonomy). 
Environments and code frameworks may differ among different branches. Thus, please follow the installation guidance provided in the specific branch you are insterested in.


## An Overall View of Research Works in This Repo  


| Category | Method | Is Contained | Is ReadME Prepared | Author | Publication | Link |
| ------ | ------ | --- | --- | ------ | ------ | ------ |
| Action | HyAR |✅ | ✅  |  Boyan Li |ICLR 2022 | https://openreview.net/forum?id=64trBbOhdGU |
| Policy | PPO-PeVFA | ✅ | ✅ | Hongyao Tang  |AAAI 2022 | https://arxiv.org/abs/2010.09536 |
| Policy | Deep SARSA-PeVFA |❌ |  ❌ | Zhentao Tang|In progress | N/A |
| Policy | TD3-PeVFA |❌ |  ❌ | Min Zhang |In progress | N/A |
| Env&task | CCM | ✅ | ❌ |Haotian Fu | AAAI 2021 | https://ojs.aaai.org/index.php/AAAI/article/view/16914 |
| Env&task | PAnDR |✅ |  ✅ |Tong Sang|In progress | N/A |
| Other | VDFP |✅ | ✅ |Hongyao Tang| AAAI 2021 | https://ojs.aaai.org/index.php/AAAI/article/view/17182 |


## TODO

- [ ] Update a liscence
- [ ] Update the README files for each branches
- [ ] Check the vadality of codes to release

## Citation

If you use our repo in your work, we ask that you cite our **paper**. 

Here is an example BibTeX:
```
@article{aaa22xxxx,
  author    = {tjurllab},
  title     = {A Unified Repo for Self-supervised RL with Representations},
  year      = {2022},
  url       = {http://arxiv.org/abs/xxxxxxx},
  archivePrefix = {arXiv}
}
```

## Liscense

**[To change]**

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.

## Acknowledgement

**[To add some acknowledgement]**


## *Update Log
2022-02-28:
- Codes for PAnDR is uploaded by @Jinyi Liu.

2022-02-26:
- Codes for CCM and VDFP is uploaded by @Hongyao Tang.

2022-02-19:
-  Codes for PeVFA-PPO and README is uploaded by @Hongyao Tang.

2022-02-16:
-  Codes for HyAR is uploaded by @Boyan Li.

2022-01-30:  
-  Repo is created and categories/folders are created.