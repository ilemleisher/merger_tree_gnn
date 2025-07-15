import numpy as np
import matplotlib.pyplot as plt
import h5py, os, optuna, torch
from scipy.spatial import KDTree
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric as torchg
import torch_scatter as torchs
import sys

def get_params(file):
    """
    This function reads in the simulation parameters for all DREAMS sims and converts them into physical units
    
    Inputs
     - file - absolute or relative path to the file containing all simulation parameters
     
    Returns
     - params - an Nx4 ndarray with the following parameters
                index 0 - WDM particle mass [keV]
                index 1 - SN1 (ew) wind energy
                index 2 - SN2 (kw) wind velocity
                index 3 - AGN1 (BHFF) black hole feedback factor
    """
    sample = np.loadtxt(file)
    
    fiducial = np.array([1,3.6,7.4,.1]) #fiducial TNG values
    params = sample * fiducial
    params[:,0] = 1/params[:,0]
    
    return params

def norm_params(params):
    """
    This function normalizes the four simulation parameters (WDN, SN1, SN2, AGN).
    
    Inputs
     - params - an Nx4 array of simulation parameters
    
    Results
     - nparams - same as the input but now normalized and linearly sampled between 0 and 1
    """
    nparams = params / np.array([1, 3.6, 7.4, .1])
    nparams[:,0] = 1/nparams[:,0]
    
    nparams[:,1:] = np.log10(nparams[:,1:])

    minimum = np.array([1/30, np.log10(0.25), np.log10(0.5), np.log10(0.25)])
    maximum = np.array([1/1.8, np.log10(4.0), np.log10(2.0), np.log10(4.0)])

    nparams = (nparams - minimum)/(maximum - minimum)

    return nparams

def denormalize(true, pred, err, type="WDM"):
    """
    This funciton denormalizes the results from the trained GNN which predicts the WDM particle mass
    
    Inputs
     - true - the correct 1/WDM particle mass from the simulation
     - pred - the predicted 1/WDM particle mass from the GNN
     - err  - the predicted 1/WDM error from the GNN
     
    Returns
     - ntrue - the denormalized true array
     - npred - the denormalized pred array
     - nerr  - the denormalized err array
    """
    if type == "WDM":
        mi = 1/30
        ma = 1/1.8
        ntrue = mi + true*(ma - mi)
        npred = mi + pred*(ma - mi)
        nerr = err*(ma-mi)

    elif type == "SN1":
        mi = np.log10(0.25)
        ma = np.log10(4.0)
        ntrue = mi + true*(ma - mi)
        npred = mi + pred*(ma - mi)
        nerr = err*(ma-mi)

        
    elif type == "SN2":
        mi = np.log10(0.5)
        ma = np.log10(2.0)
        ntrue = mi + true*(ma - mi)
        npred = mi + pred*(ma - mi)
        nerr = err*(ma-mi)

        
    elif type == "AGN":
        mi = np.log10(0.25)
        ma = np.log10(4.0)
        ntrue = mi + true*(ma - mi)
        npred = mi + pred*(ma - mi)
        nerr = err*(ma-mi)
    
    return ntrue, npred, nerr


## Creating dataset from loaded data
def create_dataset(cat, params):
    """
    This function turns DREAMS catalogs into pytorch_geometric Data objects which contain the node, edge, and graph data and the correct parameter values for each graph.
    These Data objects will then be used to create pytorch_geometric DataLoader objects that will load the data during training.

    Inputs
     - cat - a dictionary made from the DREAMS simulations containing positions and desired node features
     - rl  - linking length normalized to boxsize
     - params - an array of correct parameter values that will be used during training

    Results
     - graph - a pytorch_geometric Data object containing the node, edge, graph, and parameter values
    """
    
    # Required info
    snap = cat['SnapNum']
    subid = cat['SubhaloID']
    desid = cat['DescendantID']

    nodes = []
    for arg in cat:
        if arg != 'DMMass' and arg != 'StellarMass' and arg != 'GasMass':
            nodes.append(cat[arg])
        else:
            if arg == 'DMMass':
                nodes.append([i[1] for i in cat['SubhaloMassType']])
            if arg == 'StellarMass':
                nodes.append([i[4] for i in cat['SubhaloMassType']])
            if arg == 'GasMass':
                nodes.append([i[0] for i in cat['SubhaloMassType']])
    
    nodes_tuple = tuple(nodes)

    data = np.column_stack(nodes_tuple) #Node features associated with each node in the graph
    x = torch.tensor(data, dtype=torch.float32)
    if len(nodes_tuple) == 1:
        x = x.T
    
    ngal = len(snap)
    u = np.log10(ngal).reshape(1,1)
    u = torch.tensor(u, dtype=torch.float32)

    y = np.reshape(params, (1,params.shape[0]))
    y = torch.tensor(y, dtype=torch.float32)

    edge_index = get_edges(subid,desid)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    edge_attr=np.ones((edge_index.shape[1],3))
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    
    graph = Data(x=x, y=y, u=u, edge_index=edge_index,edge_attr=edge_attr)

    return graph


def get_edges(subid,desid):

    """
    A function taken from Cosmographnet to create edges between a pair of nodes if their distance is less than the linking radius

    Inputs
     - pos - list of positions in kpc for each node in the graph
     - rl  - linking radius normalized by boxsize

    Returns
     - edge_index - Nx2 array of indecies which point to a pair nodes connected by an edge
     - edge_attr  - Nx3 array of the edge values for each edge in the graph
    """
    #get edges

    start_edges = []
    end_edges = []
    for i in range(len(subid)):
        for j in range(len(desid)):
            if subid[i] == desid[j]:
                start_edges.append([i])
                end_edges.append([j])
    
    edges = []
    edges.append(start_edges)
    edges.append(end_edges)
    
    edge_index = np.array(edges)

    #reformat for pytorch
    edge_index = edge_index.reshape((2,-1)).astype(int)

    edge_index = edge_index.astype(int)

    return edge_index

def split_dataset(dataset, train_size, valid_size, test_size):
    """
    This function splits the simulations into training, validation, and testing sets. 
    The data are split at the simulation level so that the GNN cannot learn part of the parameter space it is tested on.
    
    Inputs:
     - dataset - pytorch_geometric Data objects; one for each simulation
     - train_size - the fractional proportion of training data (0,1) #most of data goes here, used to train nn
     - valid_size - the fractional proportion of validation data (0,1) #checking how well the nn is doing
     - test_size  - the fractional proportion of testing data (0,1) #data that gnn has never seen before, not part of training set. Makes prediction for each graph
     
    Returns:
     - train_dataset - pytorch_geometric Data objects randomly selected to be in the training set
     - valid_dataset - pytorch_geometric Data objects randomly selected to be in the validation set
     - test_dataset  - pytorch_geometric Data objects randomly selected to be in the testing set
    """
    np.random.shuffle(dataset)
    
    ndata = len(dataset)
    split_valid = int(np.floor(valid_size * ndata))
    split_test = split_valid + int(np.floor(test_size * ndata))
    
    valid_dataset = dataset[:split_valid]
    test_dataset = dataset[split_valid:split_test]
    train_dataset = dataset[split_test:]
    
    return train_dataset, valid_dataset, test_dataset
