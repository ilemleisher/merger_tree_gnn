# Merger Tree Graph Neural Network 

## Overview
This repository contains the code used in Leisher et al. in prep. to train a Graph Neural Network (GNN) on merger tree histories taken from a cosmological simulation suite within DREAMS. The codes first load subhalo data features from merger trees in Milky Way zoom-in simulation boxes in the DREAMS database. Then this data is converted into a Graph object using [PyTorch](https://pytorch-geometric.readthedocs.io/en/latest/), and finally a GNN is trained on these Graphs to infer simulation parameters.

Here is a brief description of the codes in this repository:
- ``` Merger_Tree_Script.py ```: loads data features from subhalos in the merger tree of a given box and saves them in a container.
- ``` Graph_Functions.py ```: defines functions to create the PyTorch Graphs and prepare the data for the GNN.
- ``` GNN_example.ipynb ```: example notebook for using the codes and training the GNN. Contains adapted code from [PabloVD's CosmoGraphNet](https://github.com/PabloVD/CosmoGraphNet). Creates and trains a GNN on selected data from the merger tree containers across all boxes. For more details regarding the architecture and training process, I highly recommend looking at CosmoGraphNet.
- ``` WDM_TNG_MW_SB4_parameters.txt ```: cosmological and astrophysical parameters from the SB4 WDM MW zoom-in simulation suite.
- ``` GNN_Script.py```: script that loads merger tree data from ```Merger_Tree_Script.py``` and trains a GNN with specified node features from the merger trees. Formatted to be ran via Slurm from ```Slurm_Script.sh```. Line 602 must be edited.
- ```Slurm_Script.sh```: script that creates and submits a Slurm job to train a GNN via ```GNN_Script.py```. 

## Required Dependencies
Here are the libraries required for running the codes:
- ```numpy```
- ```matplotlib```
- ```h5py```
- ```optuna```
- ```torch```
- ```scipy```
- ```torch_geometric```
- ```torch_scatter```

## Note
These scripts are all set up to be run specifically on the Rivanna HPC cluster within torrey-group, with data from the DREAMS SB4 WDM MW zoom-in simulation suite. These scripts could be adapted to run either on different suites within DREAMS, different simulations altogether, or with different HPCs, but in general, they are not meant to be used without HPC or simulation suites. Please feel free to reach out to ilemleisher@gmail.com with any questions about usage or adapting the codes.

If you use this code, please link this repository and cite Leisher et al. in prep.
