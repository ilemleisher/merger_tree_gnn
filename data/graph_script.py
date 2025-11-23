import numpy as np
import h5py, os, optuna, torch, pickle
from torch_geometric.data import Data
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
    This function denormalizes the results from the trained GNN which predicts the parameter values
    
    Inputs
     - true - the correct parameter values from the simulation
     - pred - the predicted parameter values from the GNN
     - err  - the predicted parameter values from the GNN
     
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
    These Data objects can then be used to create pytorch_geometric DataLoader objects that will load the data during training.

    Inputs
     - cat - the raw merger tree dataset for a single simulation box
     - params - an array of correct parameter values that will be used during training

    Results
     - graph - a pytorch_geometric Data object containing the node, edge, graph, and parameter values
    """
    
    # Required info for creating graphs
    snap = cat['SnapNum']
    subid = cat['SubhaloID']
    desid = cat['DescendantID']

    nodes = []

    for key in sys.argv[4:]:
        if type(cat[key][0]) == np.ndarray:
            for dim in range(len(cat[key][0])):
                nodes.append([i[dim] for i in cat[key]])
        else:
            nodes.append(cat[key])
    
    nodes_tuple = tuple(nodes)

    data = np.column_stack(nodes_tuple) #Node features associated with each node in the graph
    x = torch.tensor(data, dtype=torch.float32)
        
    ngal = len(snap)
    u = np.log10(ngal).reshape(1,1)
    u = torch.tensor(u, dtype=torch.float32)

    y = np.reshape(params, (1,params.shape[0]))
    y = torch.tensor(y, dtype=torch.float32)

    edge_index = get_edges(subid,desid,test)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    edge_attr=np.ones((edge_index.shape[1],3))
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    
    graph = Data(x=x, y=y, u=u, edge_index=edge_index,edge_attr=edge_attr)

    return graph


def get_edges(subid,desid,test="normal"):

    """
    A function taken from Cosmographnet to create edges between a pair of nodes if their distance is less than the linking radius

    Inputs
     - subid - list of subhalo ids from a single merger tree
     - desid - list of subhalo descendant ids from a single merger tree
     - test - defines the type of edges you want to draw. Defaults to "normal", which is edges directed backwards in time.

    Returns
     - edge_index - Nx2 array of indecies which point to a pair nodes connected by an edge
    """


    start_edges = []
    end_edges = []

    if test == "flattened_tree": # Draw arbitrary edges (for flattened tree test)
        for i in range(len(subid)):
            if i != len(subid) and i != len(subid)-1:
                start_edges.append([i])
                end_edges.append([i+1])
        
    elif test == "normal": # Draw edges directed backwards in time
        desid_to_idx = {}
        for j, did in enumerate(desid):
            desid_to_idx[did] = j
        
        for i, sid in enumerate(subid):
            if sid in desid_to_idx:
                start_edges.append(i)
                end_edges.append(desid_to_idx[sid])
    
    
    edges = []
    edges.append(start_edges)
    edges.append(end_edges)
    
    edge_index = np.array(edges)

    #reformat for pytorch
    edge_index = edge_index.reshape((2,-1)).astype(int)

    edge_index = edge_index.astype(int)

    return edge_index

if __name__ == "__main__":
    
    # Load in data
    with open(sys.argv[1], 'rb') as f:
        catalogs = pickle.load(f)
    
    # path to parameter file
    param_path = sys.argv[3]
    params = []
    boxes = range(len(catalogs))
    for box in boxes:
        try:
            param = get_params(param_path)[box]
            params.append(param)
        except:
            print(box)
    
    params = np.array(params)
    nparams = norm_params(params)
  
    if sys.argv[2] == 'WDM':
        params = nparams[:,0:1]
    elif sys.argv[2] == 'SN1':
        params = nparams[:,1:2]
    elif sys.argv[2] == 'SN2':
        params = nparams[:,2:3]
    elif sys.argv[2] == 'AGN':
        params = nparams[:,3:4]

    test = "normal" # Chooses which edges to draw. Can be "normal" or "flattened_tree".
    
    # Create dataset
    dataset = []
    for i in range(len(catalogs)):
        dataset.append(create_dataset(catalogs[i], params[i]))

    with open("merger_tree_graph_data.pkl", 'wb') as f:
        pickle.dump(dataset, f)
