# Behaviour-aware clustering for offline policy learning
Official implementation of paper: [URL]

## Data
Our datasets can be downloaded from [THIS LINK]. This link provides multi-behavior datasets with labels, which serve as the ground truth for evaluating clustering results. Additionally, it includes the policy trained using stable-baselines3 for generating the multi-behavior datasets.

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
python main.py --exp-name "<name>" --raw-dataset-path "<local path of multi-behaviour dataset>" --save-path "<local path>"
```
This command only contains a part of the hyperparameters you should set to run the experiments. The rest of the 

## Cite this work
```
@article{}
```
