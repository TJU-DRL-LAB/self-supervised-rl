# AAAI2021-VDFP

Source code and raw data of learning curves for AAAI 2021 paper - 《[Foresee then Evaluate: Decomposing Value Estimation with Latent Future Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/17182)》



## Description  

The source code mainly contains:  
-  implementation of our algorithm (VDFP) and other benchmark algorithms used in our experiments;  
-  the raw learning curves data and plot code.  

All the implementation and experimental details mentioned in our paper and the Supplementary Material can be found in our codes.  
  
  
## Installation

# Environment Setup
We conduct our experiments on [MuJoCo](https://roboti.us/license.html) continuous control tasks in [OpenAI gym](http://gym.openai.com). Our codes are implemented with **Python 3.6** and **Tensorflow 1.8**. (Now MuJoCo is opensource due to the proposal of DeepMind.)

# Equipment
We run our experiments on both **Windows 7** and **Ubuntu 16.04 LTS** operating systems.  

# Dependency
Main dependencies and versions are listed below:  

<div align=center>
| Dependency | Version |
| ------ | ------ |
| gym | 0.9.1 |
| mujoco-py | 0.5.7 | 
| mjpro | mjpro131 | 
| tensorflow | 1.8.0 | 
| tensorboard | 1.8.0 |
| scipy | 1.2.1 | 
| scikit-learn | 0.20.3 | 
| matplotlib | 3.0.3 | 
</div>
  
  
## Examples  
  
Examples of run commands can be seen in the file below:
> ./run/run_vdfp.sh

## Citation
If you use our repo in your work, please cite the following: 

Here is an example BibTeX:
```
@inproceedings{Tang2021VDFP,
  author    = {Hongyao Tang and
               Zhaopeng Meng and
               Guangyong Chen and
               Pengfei Chen and
               Chen Chen and
               Yaodong Yang and
               Luo Zhang and
               Wulong Liu and
               Jianye Hao},
  title     = {Foresee then Evaluate: Decomposing Value Estimation with Latent Future
               Prediction},
  booktitle = {Thirty-Fifth {AAAI} Conference on Artificial Intelligence, {AAAI}
               2021, Thirty-Third Conference on Innovative Applications of Artificial
               Intelligence, {IAAI} 2021, The Eleventh Symposium on Educational Advances
               in Artificial Intelligence, {EAAI} 2021, Virtual Event, February 2-9,
               2021},
  pages     = {9834--9842},
  publisher = {{AAAI} Press},
  year      = {2021},
  url       = {https://ojs.aaai.org/index.php/AAAI/article/view/17182},
  timestamp = {Fri, 19 Nov 2021 10:30:41 +0100},
  biburl    = {https://dblp.org/rec/conf/aaai/TangMCCCYZLH21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
