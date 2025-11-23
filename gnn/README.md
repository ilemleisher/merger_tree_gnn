# Inference Models
These codes can be used to train a GNN or XGBoost model for inferring parameter values from a merger tree dataset.
### ```gnn_script.py```:
- This script creates and trains a GNN on the merger tree dataset created with the scripts in ```data```. The GNN is trained to infer the parameter value chosen for the dataset (each graph has one parameter value). The script will first train the GNN with default hyperparameters for 1000 epochs, and then it will tune hyperparameters using Optuna for 50 trials. Note: with the current setup, and assuming the dataset has ~1000 graphs, this will take ~70 hours on a GPU (see ```example_slurm_script.sh``` for exact specifications). This script will also create three subdirectories, ```Models```, ```Outputs```, and ```Databases``` to use during training. The number of epochs and trials of hyperparameter tuning can easily be modified in the code. The default name is ```test_gnn```, but this should be changed for each time you run the script. You should use a unique name with each setup. 
- To run: ```python gnn_script.py <path/to/dataset>```
- ```<path/to/dataset>```: the path to the merger tree graph dataset. If the scripts in ```data``` are run unchanged, this should be called ```merger_tree_graph_data.pkl``` and live in the ```data``` directory.
- Outputs: The script will print the best trial (based on lowest loss) after 50 trials of hyperparameter tuning. All outputs are saved to the ```Outputs``` subdirectory that is generated with the script.
### ```xgboost.py```:
- This script trains an XGBoost model to predict parameter values with subhalo features and produce a feature importance plot. It is trained on chosen subhalo features and conducts a random search to train hyperparameters. This script can be ran quickly on a CPU. It should be modified to add or remove the subhalo features you want. Note: there is a build-in cutoff on DM mass and snapshot number.
- To run: ```python xgboost.py <path/to/dataset> <path/to/parameter/file>```
- ```<path/to/dataset>```: the path to the raw merger tree dataset (before creating graphs; this is the output of ```merger_tree_script.py```). If the scripts in ```data``` are run unchanged, this should be ```raw_merger_tree_data.pkl```.
- ```<path/to/parameter/file>```: the path to the parameter file. If unchanged, should be ```WDMM_TNG_MW_SB4_parameters.txt```.
- Outputs: Prints the R-squared value of the best performance, and creates a normalized feature importance plot.
### ```example_slurm_script.sh```:
- An example Slurm job submission file for training the GNN with default epochs, trials, and dataset. 
