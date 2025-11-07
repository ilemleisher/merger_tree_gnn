import numpy as np
import matplotlib.pyplot as plt
import h5py, os, optuna, torch, pickle
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
    for arg in sys.argv[3:]:
        if arg != 'DMMass' and arg != 'StellarMass':
            nodes.append(cat[arg])
        else:
            if arg == 'DMMass':
                nodes.append([i[0] for i in cat['SubhaloMassType']])
            if arg == 'StellarMass':
                nodes.append([i[4] for i in cat['SubhaloMassType']])
    
    nodes_tuple = tuple(nodes)

    
    data = np.column_stack(nodes_tuple) #Node features associated with each node in the graph
    x = torch.tensor(data, dtype=torch.float32)
        
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

## Hyperparameters

class Hyperparameters():
    """
    This object acts as a container for the hyperparameters that are used during training.
    This object is also used to name files that are stored during training and testing.
    """
    def __init__(self, lr, wd, nl, hc, rl, ne, name):
        
        self.learning_rate = lr
        self.weight_decay = wd
        self.n_layers = nl
        self.hidden_channels = hc
        self.r_link = rl
        self.n_epochs = ne #set small at first
        self.study_name = name
        self.outmode = 'cosmo'
        self.pred_params = 1
        
    def __repr__(self):
        return f"lr {self.learning_rate:.2e}; wd {self.weight_decay:.2e}; nl {self.n_layers}; hc {self.hidden_channels}; rl {self.r_link:.2e}"
    
    def name_model(self):
        return f"{name}_lr_{self.learning_rate:.2e}_wd_{self.weight_decay:.2e}_nl_{self.n_layers}_hc_{self.hidden_channels}_rl_{self.r_link:.2e}"

## GNN

class EdgeModel(torch.nn.Module):
    def __init__(self, node_in, node_out, edge_in, edge_out, hid_channels, residuals=True, norm=False):
        super().__init__()

        self.residuals = residuals
        self.norm = norm

        layers = [torch.nn.Linear(node_in*2 + edge_in, hid_channels),
                  torch.nn.ReLU(),
                  torch.nn.Linear(hid_channels, edge_out)]
        
        if self.norm:  
            layers.append(torchg.nn.LayerNorm(edge_out))

        self.edge_mlp = torch.nn.Sequential(*layers)


    def forward(self, src, dest, edge_attr, u, batch):

        out = torch.cat([src, dest, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.residuals:
            out = out + edge_attr
        return out

class NodeModel(torch.nn.Module):
    def __init__(self, node_in, node_out, edge_in, edge_out, hid_channels, residuals=True, norm=False):
        super().__init__()

        self.residuals = residuals
        self.norm = norm

        layers = [torch.nn.Linear(node_in + 3*edge_out + 1, hid_channels),
                  torch.nn.ReLU(),
                  torch.nn.Linear(hid_channels, node_out)]
        
        if self.norm:  
            layers.append(torchg.nn.LayerNorm(node_out))

        self.node_mlp = torch.nn.Sequential(*layers)

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = edge_attr

        # Multipooling layer
        out1 = torchs.scatter_add(out, col, dim=0, dim_size=x.size(0))
        out2 = torchs.scatter_max(out, col, dim=0, dim_size=x.size(0))[0]
        out3 = torchs.scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out1, out2, out3, u[batch]], dim=1)

        out = self.node_mlp(out)
        if self.residuals:
            out = out + x
        return out

class GNN(torch.nn.Module):
    def __init__(self, node_features, n_layers, hidden_channels, linkradius, dim_out, only_positions, residuals=True):
        super().__init__()

        self.n_layers = n_layers
        self.link_r = linkradius
        self.dim_out = dim_out
        self.only_positions = only_positions

        # Number of input node features (0 if only_positions is used)
        node_in = node_features
        # Input edge features: |p_i-p_j|, p_i*p_j, p_i*(p_i-p_j)
        edge_in = 3
        node_out = hidden_channels
        edge_out = hidden_channels
        hid_channels = hidden_channels
        
        layers = []

        # Encoder graph block
        inlayer = torchg.nn.MetaLayer(node_model=NodeModel(node_in, node_out, edge_in, edge_out, hid_channels, residuals=False),
                                      edge_model=EdgeModel(node_in, node_out, edge_in, edge_out, hid_channels, residuals=False))

        layers.append(inlayer)

        # Change input node and edge feature sizes
        node_in = node_out
        edge_in = edge_out

        # Hidden graph blocks
        for i in range(n_layers-1):

            lay = torchg.nn.MetaLayer(node_model=NodeModel(node_in, node_out, edge_in, edge_out, hid_channels, residuals=residuals),
                                      edge_model=EdgeModel(node_in, node_out, edge_in, edge_out, hid_channels, residuals=residuals))
            layers.append(lay)

        self.layers = torch.nn.ModuleList(layers)

        # Final aggregation layer
        self.outlayer = torch.nn.Sequential(torch.nn.Linear(3*node_out+1, hid_channels),
                              torch.nn.ReLU(),
                              torch.nn.Linear(hid_channels, hid_channels),
                              torch.nn.ReLU(),
                              torch.nn.Linear(hid_channels, hid_channels),
                              torch.nn.ReLU(),
                              torch.nn.Linear(hid_channels, self.dim_out))

    def forward(self, data):

        h, edge_index, edge_attr, u, batch = data.x, data.edge_index, data.edge_attr, data.u, data.batch

        # Message passing layers
        for layer in self.layers:
            h, edge_attr, _ = layer(h, edge_index, edge_attr, u, data.batch)

        # Multipooling layer
        addpool = torchg.nn.global_add_pool(h, batch)
        meanpool = torchg.nn.global_mean_pool(h, batch)
        maxpool = torchg.nn.global_max_pool(h, batch)
    
        out = torch.cat([addpool,meanpool,maxpool,u], dim=1)

        # Final linear layer
        out = self.outlayer(out)

        return out

def train_model(model, train_loader, valid_loader, hparams):
    """
    This is the main loop for training the GNN. For each epoch, the GNN is given data from the training set to update its parameters and is then tested on the validation set to see if the model has improved.
    If the model has improved, the GNN is saved in a file which can be reloaded later.
    
    Inputs
     - model - the instantiated and untrained GNN
     - train_loader - a pytorch_geometric DataLoader object containing the training dataset
     - valid_loader - a pytorch_geometric DataLoader object containing the validation dataset
     - hparams - a Hyperparameters object containing the hyperparameters to be used in this training session
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=hparams.learning_rate, max_lr=1.e-3, cycle_momentum=False, step_size_up=500)
    
    train_losses, valid_losses = [], []
    valid_loss_min, err_min = 1000., 1000.
    
    for epoch in range(1, hparams.n_epochs+1):
        train_loss = train(train_loader, model, hparams, optimizer, scheduler)
        valid_loss, err = test(valid_loader, model, hparams)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # Save model if it has improved
        if valid_loss <= valid_loss_min:
            torch.save(model.state_dict(), "Models/"+hparams.name_model())
            valid_loss_min = valid_loss
            err_min = err
            print(f"Epoch {epoch:03d} Train loss {train_loss:.2e} Valid loss {valid_loss:.2e} Error: {err:.2e} (B)")
        else:
            print(f"Epoch {epoch:03d} Train loss {train_loss:.2e} Valid loss {valid_loss:.2e} Error: {err:.2e}")
            
    return train_losses, valid_losses

def train(loader, model, hparams, optimizer, scheduler):
    """
    This function loops over all data in the training dataset, calculates the loss, and updates the GNN parameters appropriately.
    
    Inputs
     - loader - a pytorch_geometric DataLoader object containing the training dataset
     - model - the partially trained GNN object
     - hparams - a Hyperparameters object containing the hyperparameters to be used in this training session
     - optimizer - the pytorch optimizer used to update the GNN
     - scheduler - the pytorch scheduler used to vary the training rates / momentum
     
    Returns
     - loss - the average log loss from this epoch
    """
    model.train()

    loss_tot = 0
    
    for data in loader:  # Iterate in batches over the training dataset.

        data.to(device)
        optimizer.zero_grad()  # Clear gradients.
        out = model(data)  # Perform a single forward pass.
        
        y_out, err_out = out[:,:hparams.pred_params], out[:,hparams.pred_params:2*hparams.pred_params]     # Take mean and standard deviation of the output
        
        # Compute loss as sum of two terms for likelihood-free inference
        loss_mse = torch.mean(torch.sum((y_out - data.y)**2., axis=1) , axis=0)
        loss_lfi = torch.mean(torch.sum(((y_out - data.y)**2. - err_out**2.)**2., axis=1) , axis=0)
        loss = torch.log(loss_mse) + torch.log(loss_lfi)

        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        scheduler.step()
        loss_tot += loss.item()

    return loss_tot/len(loader)

def test(loader, model, hparams):
    """
    This function loops over all data in the given (validation or testing) dataset and calculates the loss. 
    The parameters of the model are not updated in this function.
    
    Inputs
     - loader - a pytorch_geometric DataLoader object containing the validation or testing dataset
     - model - the partially trained GNN object
     - hparams - a Hyperparameters object containing the hyperparameters to be used in this training session
     
    Returns
     - loss - the average log loss from this epoch
     - errs - the average absolute error from the GNN predictions
    """
    model.eval()

    trueparams = np.zeros((0,hparams.pred_params))
    outparams = np.zeros((0,hparams.pred_params))
    outerrparams = np.zeros((0,hparams.pred_params))

    errs = []
    loss_tot = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        with torch.no_grad():

            data.to(device)
            out = model(data)  # Perform a single forward pass.

            # If cosmo parameters are predicted, perform likelihood-free inference to predict also the standard deviation
            y_out, err_out = out[:,:hparams.pred_params], out[:,hparams.pred_params:2*hparams.pred_params]     # Take mean and standard deviation of the output
            
            # Compute loss as sum of two terms for likelihood-free inference
            loss_mse = torch.mean(torch.sum((y_out - data.y)**2., axis=1) , axis=0)
            loss_lfi = torch.mean(torch.sum(((y_out - data.y)**2. - err_out**2.)**2., axis=1) , axis=0)
            loss = torch.log(loss_mse) + torch.log(loss_lfi)

            err = (y_out - data.y)#/data.y
            errs.append( np.abs(err.detach().cpu().numpy()).mean() )
            loss_tot += loss.item()

            # Append true values and predictions
            trueparams = np.append(trueparams, data.y.detach().cpu().numpy(), 0)
            outparams = np.append(outparams, y_out.detach().cpu().numpy(), 0)
            outerrparams  = np.append(outerrparams, err_out.detach().cpu().numpy(), 0)
                
    
    # Save true values and predictions
    np.save("Outputs/trues_"+hparams.name_model()+".npy",trueparams)
    np.save("Outputs/outputs_"+hparams.name_model()+".npy",outparams)
    np.save("Outputs/errors_"+hparams.name_model()+".npy",outerrparams)

    return loss_tot/len(loader), np.array(errs).mean(axis=0)

def objective(trial):
    """
    This function is given to optuna to tune hyperparameters. Given the current trial, optuna will suggest new hyperparameters for this training session.
    
    Inputs
     - trial - an optuna object containing information on the current training session
     
    Returns
     - test_loss - the log loss from the testing dataset, used to compare trials and choose hyperparameters
    """
    hparams.learning_rate = trial.suggest_float("learning_rate", 1e-7, 1e-3, log=True)
    hparams.weight_decay = trial.suggest_float("weight_decay", 1e-9, 1e-6, log=True)
    hparams.n_layers = trial.suggest_int("n_layers", 1, 5)
    hparams.hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256, 512])
    hparams.r_link = trial.suggest_int("r_link", 0, 300, log=False)
    
    dataset = []
    for i in range(len(catalogs)):
        dataset.append(create_dataset(catalogs[i], hparams.r_link/boxsize, params[i]))
        
    model = GNN(node_features=dataset[0].x.shape[1],
            n_layers=hparams.n_layers,
            hidden_channels=hparams.hidden_channels,
            linkradius=hparams.r_link,
            dim_out=len(prediction)*2,
            only_positions=False)
    
    if torch.cuda.is_available():
        device = torch.device('cuda') #gpu
    else:
        device = torch.device('cpu')
    model.to(device)
        
    train_data, valid_data, test_data = split_dataset(dataset, train_size, valid_size, test_size)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    train_losses, valid_losses = train_model(model, train_loader, valid_loader, hparams)
    
    np.save(f"Outputs/train_loss_{hparams.name_model()}", train_losses)
    np.save(f"Outputs/valid_loss_{hparams.name_model()}", valid_losses)
    
    state_dict = torch.load("Models/"+hparams.name_model(), map_location=device)
    model.load_state_dict(state_dict)
    
    test_loss, err = test(test_loader, model, hparams)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return test_loss

def objective(trial):
    """
    This function is given to optuna to tune hyperparameters. Given the current trial, optuna will suggest new hyperparameters for this training session.
    
    Inputs
     - trial - an optuna object containing information on the current training session
     
    Returns
     - test_loss - the log loss from the testing dataset, used to compare trials and choose hyperparameters
    """
    hparams.learning_rate = trial.suggest_float("learning_rate", 1e-7, 1e-3, log=True)
    hparams.weight_decay = trial.suggest_float("weight_decay", 1e-9, 1e-6, log=True)
    hparams.n_layers = trial.suggest_int("n_layers", 1, 5)
    hparams.hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256, 512])
    hparams.r_link = trial.suggest_int("r_link", 0, 300, log=False)   # can delete
    
    dataset = []
    for i in range(len(catalogs)):
        dataset.append(create_dataset(catalogs[i], params[i]))
        
    model = GNN(node_features=dataset[0].x.shape[1],
            n_layers=hparams.n_layers,
            hidden_channels=hparams.hidden_channels,
            linkradius=hparams.r_link,
            dim_out=len(prediction)*2,
            only_positions=False)
    
    if torch.cuda.is_available():
        device = torch.device('cuda') #gpu
    else:
        device = torch.device('cpu')
    model.to(device)
        
    train_data, valid_data, test_data = split_dataset(dataset, train_size, valid_size, test_size)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    train_losses, valid_losses = train_model(model, train_loader, valid_loader, hparams)
    
    np.save(f"Outputs/train_loss_{hparams.name_model()}", train_losses)
    np.save(f"Outputs/valid_loss_{hparams.name_model()}", valid_losses)
    
    state_dict = torch.load("Models/"+hparams.name_model(), map_location=device)
    model.load_state_dict(state_dict)
    
    test_loss, err = test(test_loader, model, hparams)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return test_loss

## Data to be loaded in
with open('merger_tree_data', 'rb') as f: #<--- YOUR DATA HERE. Format follows the output from Merger_Tree_Script.py
    catalogs = pickle.load(f)

# path to parameter file

param_path = 'WDM_TNG_MW_SB4_parameters.txt'
params = []
boxes = range(1024)
for box in boxes:
    try:
        param = get_params(param_path)[box]
        params.append(param)
    except:
        print(box)


params = np.array(params)
nparams = norm_params(params)
if sys.argv[1] == 'WDM':
    params = nparams[:,0:1]
    
elif sys.argv[1] == 'SN1':
    params = nparams[:,1:2]
    
elif sys.argv[1] == 'SN2':
    params = nparams[:,2:3]
    
elif sys.argv[1] == 'AGN':
    params = nparams[:,3:4]

## Create dataset
h = 0.6909
dataset = []
for i in range(len(catalogs)):
    dataset.append(create_dataset(catalogs[i], params[i]))

train_size = 0.8 #standard parameters
valid_size = 0.1
test_size  = 0.1
batch_size = 32

train_data, valid_data, test_data = split_dataset(dataset, train_size, valid_size, test_size)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True) 
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

base_name = ""+str(sys.argv[1])
for arg in sys.argv[3:]:
    base_name += "_"+str(arg)
base_name += "_"+str(sys.argv[2])
name = base_name              #name that files will be saved as. Should be different for every unique GNN that you train
boxes = range(1024)        #which simulations you include, there are 1024 simulations
snap = 90                  #which redshift you use, right now the script only works for snap >~ 60
h = 0.6909                 #reduced hubble factor, don't change this
dm_mass_thresh = 1e8       #only include satellites above this halo mass
stellar_mass_thresh = 0  #only include satellites above this stellar mass
boxsize = 100000 / h       #box size, don't change this
prediction = [0]   

rl = 300 #Might need to be changed
lr = 7e-4      #learning rate
wd = 2e-8      #weight decay
nl = 2         #number of layers
hc = 512       #hidden channels (power of 2)
n_epochs = 1000  #number of epochs
hparams = Hyperparameters(lr, wd, nl, hc, rl, n_epochs, name)
print(hparams)

## Create GNN based on hyperparameters
model = GNN(node_features=dataset[0].x.shape[1],
            n_layers=hparams.n_layers,
            hidden_channels=hparams.hidden_channels,
            linkradius=hparams.r_link,
            dim_out=len(prediction)*2,
            only_positions=False)

#Move to gpu/cpu
if torch.cuda.is_available():
    device = torch.device('cuda') #gpu
else:
    device = torch.device('cpu')
model.to(device)
print(device)

train_losses, valid_losses = train_model(model, train_loader, valid_loader, hparams)

state_dict = torch.load("Models/"+hparams.name_model(), map_location=device)
model.load_state_dict(state_dict)

test_loss, err = test(test_loader, model, hparams)
valid_loss = np.min(valid_losses)
train_loss = train_losses[np.argmin(valid_losses)]
print(train_loss, valid_loss, test_loss)
    
#Hyperparameter tuning: multiple trials

name = base_name+"_trials"
storage = f"sqlite:///{os.getcwd()}/Databases/optuna_{name}"
n_trials = 50
sampler = optuna.samplers.TPESampler(n_startup_trials=n_trials//3)
study = optuna.create_study(study_name=name, sampler=sampler, storage=storage, load_if_exists=True)

study.optimize(objective, n_trials, gc_after_trial=True)

trials = study.trials
losses = [el.value for el in trials]
print(losses)

trials = study.trials
losses = [el.value for el in trials]
print([x for x in losses if x != None])
best_idx = np.argsort([x for x in losses if x != None])[0]
best_trial = trials[best_idx]
print(best_trial)

hparams.learning_rate = best_trial.params["learning_rate"]
hparams.n_layers = best_trial.params["n_layers"]
hparams.hidden_channels = best_trial.params["hidden_channels"]
hparams.r_link = best_trial.params["r_link"]
hparams.weight_decay = best_trial.params['weight_decay']

outputs = np.load("Outputs/outputs_"+hparams.name_model()+".npy")
trues = np.load("Outputs/trues_"+hparams.name_model()+".npy")
errors = np.load("Outputs/errors_"+hparams.name_model()+".npy")

graph_dict = dict(trues = trues, outputs = outputs, errors = errors)

true, pred, err = denormalize(trues, outputs, errors, type=sys.argv[1])
true = true[:,0]
pred = pred[:,0]
err = np.abs(err[:,0])
graph_dict = dict(true = true, pred = pred, err = err)

with open(base_name+"_graph",'wb') as f:
    pickle.dump(graph_dict, f)

print(name)