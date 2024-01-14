# Behaviour-aware clustering for offline policy learning
Official implementation of paper: [URL]

## Download datasets
Our datasets can be downloaded from [THIS LINK](https://drive.google.com/drive/folders/14EYcggpa4KCgRevSe0dh3H3-8leQGIok). This link provides multi-behavior datasets with labels, which serve as the ground truth for evaluating clustering results. Additionally, it includes the policy trained using [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/) for generating the multi-behavior datasets. It should be emphasized that all datasets include observations, actions, rewards, terminals, and labels, making them suitable for training policies as well.

## Using our code
### Installation
Our project can be installed by cloning the repository as follows:
```
https://github.com/wq13552463699/Behaviour-aware-clustering-for-offline-policy-learning.git
```
Or you can download this GitHub project and unzip it locally. 

Then you can install the required libraries by running:  
```
pip install -r requirements.txt
```

### Run experiments
You can run the experiments by performing:
```
python main.py --exp-name <name> --raw-dataset-path <local path of multi-behaviour dataset> --save-path <local path>
```
This command includes only a subset of the hyperparameters required to execute the experiments. You can find the remaining hyperparameters in the **main.py** file.  

After the clustering process terminates as convergence, a file **estimated_traj_labels.pkl** will be created in the specified save path. This file contains the clustering results as discrete labels, which can then be compared with the ground truth labels for evaluation.

### Pretrained results
Our pretrained results can be accessed by following [THIS LINK](https://drive.google.com/drive/folders/1OYTtaq-Y-bH3j030jGdch71eGQoJnkqr?usp=sharing), which contains the tuned hyperparameters, the clustering results, and the trained neural network models.

## Cite this work
```
@article{}
```
