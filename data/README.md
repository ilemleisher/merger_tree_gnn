# Data Preparation Scripts
These codes can use used to create merger tree PyTorch graph structures from DREAMS WDM simulation suite.
### ```merger_tree_script.py```:
- This script takes simulation boxes and "walks" the unique merger tree in each box, in the process saving data from each subhalo to a pickled output file. Note: you may have to change the path to the simulation boxes, and you may want to change the name or location of the output file. There are also 17 boxes that are always removed, since they are corrupted (as of 11/25). The corresponding entries in the parameter file are also already removed.
- To run: ```python merger_tree_script.py <first box> <last box> <data features...>```
- ```<first box>```: the first simulation box that you want to load merger tree data for.
- ```<last box>```: the last simulation box that you want to load merger tree data for. The script will load all boxes in between the first and last.
- ```<data features...>```: the subhalo data features you want to save from each subhalo along each merger tree. Follow [TNG documentation](https://www.tng-project.org/data/docs/specifications/#sec4a) for SubLink and Subfind catalogs for reference.
- Outputs: pickled ```raw_merger_tree_data.pkl``` file containing a dictionary for each simulation box, where each dictionary has keys corresponding to the data features called.
### ```graph_script.py```:
- This script takes the raw merger tree dataset and turns it into PyTorch graph objects with edges directed backwards in time. Parameter values corresponding to the parameter you choose are attached to each graph as well. Note: you may need to change the ```test``` variable in the code from ```normal``` to ```flattened_tree``` if you want to generate the flattened tree structures used in Leisher et al. 2025.
- To run: ```python graph_script.py <path/to/dataset> <parameter> <path/to/parameter/file>```
- ```<path/to/dataset>```: the path to your raw merger tree dataset generated with ```merger_tree_script.py```. If you ran ```merger_tree_script.py``` unchanged, then this should just be ```raw_merger_tree_data.pkl```.
- ```<parameter>```: the parameter you want attached to your graphs. Can be ```WDM```, ```SN1```, ```SN2```, or ```AGN```. This will end up being the target parameter for GNN inference.
- ```<path/to/parameter/file>```: the path to the file containing the parameter values. If unchanged, this should just be ```WDM_TNG_MW_SB4_parameters.txt```.
- Outputs: pickled ```merger_tree_graph_data.pkl``` file containing the PyTorch graphs for each simulation box.
### ```dreams.py```:
- This is just a helper script that defines functions for the other scripts.
### ```WDM_TNG_MW_SB4_parameters.txt```
- This file contains the parameter values for each simulation box in the WDM MW SB4 DREAMS suite. There are 17 entries removed, corresponding to corrupted boxes.
