# Merger Tree Graph Neural Network 

## Overview
This repository contains the code used in Leisher et al. in prep. to train a Graph Neural Network (GNN) on merger tree histories taken from a cosmological simulation suite within DREAMS. The codes load subhalo data features from merger trees in Milky Way zoom-in simulation boxes in the DREAMS database. This data is converted into a Graph object using [PyTorch](https://pytorch-geometric.readthedocs.io/en/latest/), and then a GNN is trained on these Graphs to infer simulation parameters.

Here is a brief description of the codes in this repository:
- ``` WalkingMergerTree.py ```: loads data features from subhalos in the merger tree of a given box and saves them in a container.
- ``` GNN_script.py ```: adapted script from [PabloVD's CosmoGraphNet](https://github.com/PabloVD/CosmoGraphNet). Creates and trains a GNN on selected data from the merger tree containers across all boxes.
- ``` gnn_tutorial.ipynb ```: tutorial notebook for using both the scripts.

