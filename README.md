<div align="center">
  <a href="http://erl.ucsd.edu/">
    <img align="left" src="docs/static/images/erl.png" width="80" alt="erl">
  </a>
  <a href="https://contextualrobotics.ucsd.edu/">
    <img align="center" src="docs/static/images/cri.png" width="150" alt="cri">
  </a>
  <a href="https://ucsd.edu/">
    <img align="right" src="docs/static/images/ucsd.png" width="260" alt="ucsd">
  </a>
</div>

# Planning Neural Operator: Generalizable motion planning via operator learning
<div align="center">
 <a href="#"><img alt="Main result from the paper" src="docs/static/images/mainResult.png" width="100%"/></a>
</div>


## Introduction

This repository is the official implementation of "Generalizable Motion Planning via Operator Learning". Please refer to the 2D example below for quick access into the method 
and for a futher dive, please see the models section below. 

More extensive experiments related to iGibson Environments, Manipulator and Neural Heuristics are covered in the last section

# 2D example - Ablation & Super-Resolution Results
See `examples` folder for a notebook that immediately reproduces the 2D results for the paper. To get the notebook working, 
we provide both pretrained models as well as the corresponding datasets to make your own at 
- [Dataset](https://huggingface.co/datasets/lukebhan/generalizableMotionPlanning)
- [Models](https://huggingface.co/lukebhan/generalizableMotionPlanningViaOperatorLearning)

To ensure the paths are correct, place both folders inside the example folder. The dataset is quite large and may take some time to download. 
The current example is working under Python 3.10.1, PyTorch Version 2.5.1, and the compatabile numpy libraries. Be sure to use this version
of torch in your Cuda enviornment to avoid compatabilitiy issues. 

Installation

- Setup your CUDA environment (or CPU) with the corresponding packages
  
  >The repository directory should look like this:
```
PNO/
├── examples/
│   ├── dataset/    # training and testing dataset
│   ├── results/    # training results
│   ├── utilities/    
│   ├── model/ # model implementations
│   └── 2DexampleNotebook.ipynb   # jupyternotebook
•   •   •
•   •   •
```
where the hierarchy is given via bullets and the results folder has all of the models downloaded from Hugging Face.

- Run the jupyter-notebook. Sections 1 and 2 allow you to train your own models but they can be skipped. Sections 3 and 4 quantitatively test the models and reproduce the results in the paper. 

If you have any issues, feel free to create an issue in this github repo or email the authors.


### Neural Heurisics on MovingAI lab 2D city dataset
Here we link the dataset and models used in the Neural Heuristics section. Please note that to compute the value functions, we used the Dijkstra's algorithm. For more details, please check the Appendix. Dataset and models used for each experiment are linked below.

256x256
- [Dataset](https://huggingface.co/datasets/sharathmatada/2D-256-Dataset-0)
- [Model](https://huggingface.co/sharathmatada/256-MovingAI)

512x512
- [Dataset](https://huggingface.co/datasets/sharathmatada/2D-512-Dataset-0)
- [Model](https://huggingface.co/sharathmatada/512-MovingAI)

1024x1024
- [Dataset](https://huggingface.co/datasets/sharathmatada/2D-1024-Dataset-0)
- [Model](https://huggingface.co/sharathmatada/1024-MovingAI)


[//]: # (### Citation )

[//]: # (If you found this work useful, we would appreciate if you could cite our work:)

[//]: # (```)

[//]: # (@inproceedings{)

[//]: # (matada2025generalizable,)

[//]: # (title={Generalizable Motion Planning via Operator Learning},)

[//]: # (author={Sharath Matada and Luke Bhan and Yuanyuan Shi and Nikolay Atanasov},)

[//]: # (booktitle={The Thirteenth International Conference on Learning Representations},)

[//]: # (year={2025},)

[//]: # (url={https://openreview.net/forum?id=UYcUpiULmT})

[//]: # (})

[//]: # (```)
