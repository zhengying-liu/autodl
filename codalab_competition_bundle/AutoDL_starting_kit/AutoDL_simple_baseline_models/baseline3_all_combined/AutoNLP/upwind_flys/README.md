# AutoNlp
AutoNLP-WAIC2019

AutoDL competition introduction:[NeurIPS 2019 AutoDL Challenges](https://autodl.chalearn.org/)

Team: Upwind_flys
Rank: Second place


## Methods:
Our algorithm process data and select models automatically, model lib contains Character-based model, word-based model, which can be selected according to data meta-feature. Then algorithm automatically select early stop strategy and restore weights based on the Information of feedback simulation.

## Document description:
Code Framework is [AutoNlp-WAIC2019 starting kit](https://github.com/mortal123/autonlp_starting_kit)  
AutoDL_ingestion_program/: The code and libraries used on Codalab to run your submission.  
AutoDL_scoring_program/: The code and libraries used on Codalab to score your submission.  
AutoDL_sample_code_submission/: An example of code submission you can use as template.  
AutoDL_sample_data/: Some sample data to test your code before you submit it.  

## Main python module:
run_local_test.py: A python script to simulate the runtime in codalab  
model.py: Implementation of our algorithm and logics  
data_manager.py: Data processing related module  
model_manager.py: Automatic model generation from model library  

Run the project locally: 
```
python run_local_test.py -dataset_dir=./AutoDL_sample_data/DEMO -code_dir=./AutoDL_sample_code_submission
```


## Experiment Results:

metrics  |  O1  | O2  | O3  |  O4  |  O5
---- | ----- | ------  | ----- | ----- | ----- |
 ALC | 0.8139 |  0.9277  | 0.8053 | 0.9758 | 0.8870 | 
2AUC-1  | 0.8168 | 0.9723 | 0.8345 | 0.9966 | 0.9447 |


## Other related work:
Our work in AutoML and meta-learning fields:
[Efficient Automatic Meta Optimization Search for Few-Shot Learning](https://arxiv.org/abs/1909.03817)

## Licensing
The project is developed at Lenovo Inc,It is distributed under [MIT LICENSE](https://github.com/upwindflys/AutoNlp/blob/master/LICENSE)
