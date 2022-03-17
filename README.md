# **Self-supervised RL**: A Unified Algorithmic Framework & Opensource Code Implementation of Algorithms for Self-supervised Reinforcement Leanring (SSRL) with Representations

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



### A Unified Algorithmic Framework (Implementation Design) of SSRL Algorithm

All SSRL algorithms with representation in our taxonmy follows the same algorithmic framework.  
The illsutration of our Unified Algorithmic Framework (Implementation Design) of SSRL Algorithm is shown below.
From left to right, the framework consists of four phases:
- **Data Input**
- **Encoding and Transformation**
- **Methods and Criteria of Representation Learning**
- **Downstream RL Problems**

The unified framework we propose is general. Almost all currently existing SSRL algorithms can be interpreted with our framework. 
In turn, this unified framework can also serve as a guidance when we are working on designing a new algorithm.

<div align=center>![Ecology of SSRL](https://github.com/TJU-DRL-LAB/self-supervised-rl/raw/master/assets/alg_framework.png)</div>


### Ecology of SSRL

Beyond the opensource of our research works, we plan to establish the ecology of SSRL in the future.
Driven by **three key fundamental challenges of RL**, we are working on research explorations at the frontier 
**from the different perspectives of self-supervised representation in RL**.
For algorithms and methods proposed, we plan to study **a unified algorithmic framework** togather with **a unified opensource code-level implementation framework**.
These representations are expected to **boost the learning in various downstream RL problems**, in straightforward or sophatiscated ways.
Finally, our ultimate goal is to **land self-supervised representation driven RL in real-world decision-making scenarios**.

<div align=center>![Ecology of SSRL](https://github.com/TJU-DRL-LAB/self-supervised-rl/blob/main/assets/Ecology_of_SSRL.png)</div>


## Installation

The algorithms in this repo are all implemented **python 3.5** (and versions above). 
**Tensorflow 1.x** and **PyTorch** are the main DL code frameworks we adopt in this repo with different choices in different algorithms.

First of all, we recommend the user to install **anaconada** and or **venv** for convenient management of different python envs.

In this repo, the following RL environments may be needed:
- [OpenAI Gym](https://github.com/openai/gym) (e.g., MuJoCo, Robotics)
- [MinAtar](https://github.com/kenjyoung/MinAtar)
- ......
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
| Env&task | CCM | ❌ | ❌ |Haotian Fu | AAAI 2021 | https://ojs.aaai.org/index.php/AAAI/article/view/16914 |
| Env&task | PAnDR |❌ | ❌ |Tong Sang| In progress | N/A |
| Other | VDFP |✅ | ✅ |Hongyao Tang| AAAI 2021 | https://ojs.aaai.org/index.php/AAAI/article/view/17182 |


## TODO

- [ ] Update a liscence
- [ ] Update the README files for each branches
- [ ] Check the vadality of codes to release

## Citation

If you use our repo in your work, please cite the following: 

Here is an example BibTeX:
```
@article{tjurllab22ssrl,
  author    = {TJU RL Lab},
  title     = {A Unified Repo for Self-supervised RL with Representations},
  year      = {2022},
  url       = {https://github.com/TJU-DRL-LAB/self-supervised-rl},
}
```



## *Update Log
2022-03-18:
- Main page readme uploaded.
- VDFP, HyAR, PeVFA codes - first commit.
