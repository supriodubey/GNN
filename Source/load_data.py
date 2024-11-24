# +
# Routine for loading the QUIJOTE halos LH catalogues
# -

import h5py
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from Source.constants import *
from Source.plotting import *
import scipy.spatial as SS
import readfof
import pickle
from matplotlib.ticker import ScalarFormatter
# from tqdm.notebook import tqdm
from tqdm import tqdm
from joblib import Parallel, delayed


# Normalize QUIJOTE parameters
def normalize_params(params):

    minimum = np.array([0.1,0.03,0.5,0.8,0.6,0.01,])#-600,
    maximum = np.array([0.5,0.07,0.9,1.2,1.0,1.0,]) #600,
    params = (params - minimum)/(maximum - minimum)
    return params

# def std_normalize_params(params):
#     mean_params = params.mean(axis=0)
#     std_params = params.std(axis = 0)
#     norm_params = (params - mean_params)/std_params
#     return norm_params

# KDTree: provides an index into a set of k-dimensional points 
# which can be used to rapidly look up the nearest neighbours of any point

# Compute KDTree and get edges and edge features
def get_edges(pos, r_link):

    # 1. Get edges

    # Create the KDTree and look for pairs within a distance r_link (clustering phase)

    # Boxsize normalized to 1
    kd_tree = SS.KDTree(
        pos,                # data 
        leafsize = 16,      # threshold at which the algorithm stops splitting 
                            # and directly stores points in a leaf node
                            # threshold on the number of points
        boxsize = 1.0001    # apply a m-d toroidal topology to the KDT 
                            # --> periodic boundary condition
                            # but small tolerance (small value for boxsize)
                            # --> the tree accounts for points very close 
                            # to the boundary, improving the accuracy of neighbor 
                            # and distance calculations in a periodic space
        )
    
    # Find all pairs of points in the KDT whose distance is at most r (maximum distance)
    # returns point indexes
    edge_index = kd_tree.query_pairs(r=r_link, output_type="ndarray") 

    # Ensure that for every pair of points found within r_link, 
    # the reverse pair is also included --> symmetry
    reversepairs = np.zeros((edge_index.shape[0],2))
    for i, pair in enumerate(edge_index):
        reversepairs[i] = np.array([pair[1], pair[0]])
    edge_index = np.append(edge_index, reversepairs, 0)

    # indexes must be integers
    edge_index = edge_index.astype(int)

    # Write in pytorch-geometric format
    edge_index = edge_index.T
    num_pairs = edge_index.shape[1]

    # 2. Get edge attributes
    row, col = edge_index

    # Calculating distance between linked halo pairs
    diff = pos[row]-pos[col]

    # Taking into account periodic boundary conditions
    diff_bc = np.where(diff < -0.01, diff + 1.0, diff)
    diff = np.where(diff > 0.01, diff - 1.0, diff_bc)

    # Get translational and rotational invariant features

    # Distance d = sqrt(dx^2+dy^2+dz^2)
    dist = np.linalg.norm(diff, axis=1) 

    # Centroid of halo catalogue (3d position of the centroid)
    centroid = np.mean(pos,axis=0)

    # Vectors of node and neighbor --> ??
    # distance between each point and the centroid
    row = (pos[row] - centroid)
    col = (pos[col] - centroid)

    # Taking into account periodic boundary conditions
    row_bc = np.where(row < -0.5, row + 1, row)
    row = np.where(row > 0.5, row - 1, row_bc)

    col_bc = np.where(col < -0.5, col + 1, col)
    col = np.where(col > 0.5, col - 1, col_bc)

    # Normalizing
    unitrow = row/(np.linalg.norm(row, axis = 1).reshape(-1, 1))  
    unitcol = col/(np.linalg.norm(col, axis = 1).reshape(-1, 1))
    unitdiff = diff/(dist.reshape(-1,1))

    # Dot products between unit vectors
    cos1 = np.array([np.dot(unitrow[i,:].T,unitcol[i,:]) for i in range(num_pairs)])
    cos2 = np.array([np.dot(unitrow[i,:].T,unitdiff[i,:]) for i in range(num_pairs)])

    # Normalize distance by linking radius
    dist /= r_link

    # Concatenate to get all edge attributes
    edge_attr = np.concatenate([dist.reshape(-1,1), cos1.reshape(-1,1), cos2.reshape(-1,1)], axis=1)

    # Self loops (self interactions)
    loops = np.zeros((2,pos.shape[0]),dtype=int)
    atrloops = np.zeros((pos.shape[0],3))
    for i, _ in enumerate(pos):
        loops[0,i], loops[1,i] = i, i
        atrloops[i,0], atrloops[i,1], atrloops[i,2] = 0., 1., 0.
    edge_index = np.append(edge_index, loops, 1)
    edge_attr = np.append(edge_attr, atrloops, 0)
    
    edge_index = edge_index.astype(int)

    return edge_index, edge_attr

####################################### Trial ###############################################
def compute_edges_for_chunk(chunk_idx, pos, r_link):
    """
    Helper function to compute edges for a subset of the positions (a chunk).
    """
    start, end = chunk_idx
    kd_tree = SS.KDTree(pos[start:end],leafsize = 16, boxsize=1.0001)
    edge_index = np.array(list(kd_tree.query_pairs(r=r_link)), dtype=np.int64)

    # Add reverse pairs for symmetry
    reversepairs = edge_index[:, [1, 0]]
    edge_index = np.vstack([edge_index, reversepairs]).T

    return edge_index

def get_edges_parallel(pos, r_link, num_jobs=-1):
    """
    Computes edges and edge attributes in parallel using joblib.

    Parameters:
    pos (numpy.ndarray): Positions of points.
    r_link (float): Linking radius.
    num_jobs (int): Number of parallel jobs. Use -1 for all available CPUs.

    Returns:
    edge_index (numpy.ndarray): Edge indices.
    edge_attr (numpy.ndarray): Edge attributes.
    """
    # Spliting the data into chunks for parallel processing
    chunk_size = len(pos) // (num_jobs if num_jobs != -1 else 1)
    chunks = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_jobs if num_jobs != -1 else 1)]

    # Parallelizing the edge computation
    edge_indices_list = Parallel(n_jobs=num_jobs)(delayed(compute_edges_for_chunk)(chunk, pos, r_link) for chunk in chunks)
    
    # Concatenating the results from each chunk
    edge_index = np.hstack(edge_indices_list)

    # Computing the edge attributes 
    row, col = edge_index
    diff = pos[row] - pos[col]
    
    # periodic boundary conditions
    diff = np.where(diff < -0.5, diff + 1.0, diff)
    diff = np.where(diff > 0.5, diff - 1.0, diff)

    dist = np.linalg.norm(diff, axis=1)
    centroid = np.mean(pos, axis=0)

    row_offset = pos[row] - centroid
    col_offset = pos[col] - centroid

    row_offset = np.where(row_offset < -0.5, row_offset + 1, row_offset)
    row_offset = np.where(row_offset > 0.5, row_offset - 1, row_offset)

    col_offset = np.where(col_offset < -0.5, col_offset + 1, col_offset)
    col_offset = np.where(col_offset > 0.5, col_offset - 1, col_offset)

    # Normalize vectors
    unit_row = row_offset / np.linalg.norm(row_offset, axis=1, keepdims=True)
    unit_col = col_offset / np.linalg.norm(col_offset, axis=1, keepdims=True)
    unit_diff = diff / dist[:, np.newaxis]

    # Compute cosine similarities
    cos1 = np.sum(unit_row * unit_col, axis=1)
    cos2 = np.sum(unit_row * unit_diff, axis=1)

    # Normalize distance by linking radius
    dist /= r_link

    edge_attr = np.stack([dist, cos1, cos2], axis=1)

    # Add self-loops
    num_points = len(pos)
    loops = np.arange(num_points)
    loop_attr = np.zeros((num_points, 3))
    loop_attr[:, 1] = 1

    edge_index = np.hstack([edge_index, np.vstack([loops, loops])])
    edge_attr = np.vstack([edge_attr, loop_attr])

    return edge_index, edge_attr
#############################################################################################


# Routine to create a cosmic graph from a halo catalogue
def sim_graph(simnumber, filename, paramsfile, hparams):

    # Get some hyperparameters
    r_link, pred_params, snapnum = hparams.r_link, hparams.pred_params, hparams.snap

    # Read Fof
    FoF = readfof.FoF_catalog(
        filename,           # simulation file name
        snapnum,#45,                  # snapnum, indicating the redshift (z=0)
        long_ids = False,
        swap = False,
        SFR = False,
        read_IDs = False
        )
    
    # Get positions and masses
    pos = FoF.GroupPos/1e06             # Halo positions in Gpc/h 
    mass_raw = FoF.GroupMass * 1e10     # Halo masses in Msun/h
    
    # Mass cut
    cut_val = np.quantile(mass_raw,0.997)    # universal mass cut
    mass_mask = (mass_raw >= cut_val)
    mass_mask = mass_mask.reshape(-1) # CHECK
    mass = mass_raw[mass_mask]  
    pos = pos[mass_mask]    

   
    # Get the output to be predicted by the GNN
    
    
    # Read the value of the cosmological parameters
    params = np.array(paramsfile[simnumber],dtype=np.float32)
    # print(f"parameters in load_data is:\n {params}")
    
    # Normalize true parameters
    
    # norm_params = std_normalize_params(params) # uncomment when using multiple params
    norm_params = normalize_params(params)
    # norm_params = std_normalize_params(params)
    # print(f"normalised parameters in load_data:\n {params}")
    
    # Consider the correct number of parameters
    
    parameters = norm_params[pred_params - 1: pred_params]#[pred_params - 1: pred_params] # [:pred_params]
    # print(f"Number of parameters being predicted {pred_params}\n")
    # print(f" Parameters being predicted {params}\n")

    # label
    y = np.reshape(parameters, (1,parameters.shape[0]))
    # print(f" y in load-data is: {y}\n")
    
    # Number of halos as global features
    u = np.log10(pos.shape[0]).reshape(1,1) 
    
    # Nodes features
    x = torch.tensor(mass, dtype=torch.float32)
    #?????????-----Add velocity------????????? should it be a node feature ot global?
    
    
    # Get edges and edge features
    edge_index, edge_attr = get_edges(pos, r_link)
    # edge_index, edge_attr = get_edges_parallel(pos, r_link,num_jobs = 25)
    
    graph = Data(
        x = x.resize_(x.size()[0],1),                               # node feature
        y = torch.tensor(y, dtype=torch.float32),                   # true label
        u = torch.tensor(u, dtype=torch.float32),                   # global features
        edge_index = torch.tensor(edge_index, dtype=torch.long),    # graph connectivity
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)    # edge feature matrix
        ) 
    
    return graph, cut_val

# Training and validation splitting
def split_datasets(dataset):

    random.shuffle(dataset)

    num_train = len(dataset)

    # From split percentages get indexes
    split_valid = int(np.floor(valid_size * num_train))
    split_test = split_valid + int(np.floor(test_size * num_train))

    # Splitting
    train_dataset = dataset[split_test:]
    valid_dataset = dataset[:split_valid]
    test_dataset = dataset[split_valid:split_test]
    
    # Create dataloader
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader


#Parallelizing the simgraph function in presence of multiple processor



# Main routine to load data and create the dataset   
def create_dataset(hparams, verbose=False):

    # Simulation numbers
    simnumber = np.arange(0, hparams.n_sims)
    simnumber = simnumber.astype(str).tolist()

    # True parameters
    # param_file = "/mnt/Halos/latin_hypercube_EQ/latin_hypercube_params.txt" # CHECK
    param_file = "/mnt/Halos/latin_hypercube_nwLH/latin_hypercube_params_nwlh.txt"
     # only for nwLH
    paramsfile = np.loadtxt(param_file, comments ='#')
    paramsfile = paramsfile[:, :-1]

    # Create dataset
    def process_simulation(numsim, sim, paramsfile, hparams):
        # fpath = f"/mnt/Halos/latin_hypercube_EQ/LH_{sim}"
        fpath = f"/mnt/Halos/latin_hypercube_nwLH/{sim}"
        graph, masscut = sim_graph(numsim, fpath, paramsfile, hparams)
        if verbose:
            print(f"Graph {sim} Uploaded")
        return graph, masscut

    results = Parallel(n_jobs=25)(delayed(process_simulation)(numsim, sim, paramsfile, hparams) for numsim,sim in tqdm(enumerate(simnumber)))
 
    # Separate graphs and masscuts from the results
    dataset, masscuts_list = zip(*results)
        
# Uncomment when using single CPU
    
    # for numsim, sim in tqdm(enumerate(simnumber), total=len(simnumber), desc="Processing simulations"):
        
        
        # fpath = f"/mnt/Halos/latin_hypercube_EQ/LH_{sim}"
        # graph, masscut = sim_graph(numsim, fpath, paramsfile, hparams)
        # dataset.append(graph)
        # masscuts_list.append(masscut)
    if verbose:
        print("Graph {0} Uploaded".format(numsim))



    
        
    
    
    masscuts = np.array(masscuts_list)
    quantiles = [0.025,0.5,0.975]
    quantile_points = np.quantile(masscuts, quantiles)
    mass_hist, mass_edges = np.histogram(a=masscuts, bins=80)
    bin_width = mass_edges[1] - mass_edges[0]

    # PLOT MASSCUT
    col_1 = '#648FFF'
    col_2 = '#785EF0'
    col_3 = '#DC267F'
    col_4 = '#FE6100'
    col_5 = '#FFB000'

    _, ax = plt.subplots(figsize=(10,6))
    ax.grid(alpha=0.4, linestyle='--')
    
    ax.hist(x=masscuts, bins=mass_edges, alpha = 0.5, label='Masscuts distribution', color = col_1)
    lims = plt.gca().get_ylim()
    ax.vlines(quantile_points, lims[0], lims[1], color=col_3, linestyle='--')
    maxim = mass_edges[np.argmax(mass_hist)]+bin_width*0.5
    ax.vlines(maxim, lims[0], lims[1], color=col_2)
    ax.set_ylim(lims[0],lims[1])
    for i, x in enumerate(quantile_points):
        plt.text(x-1e13, -7.5, "{0} %".format(quantiles[i]*100), rotation=0, color=col_3)
        plt.text(x-1e13, -5.2, "{0}".format(round(x/1e14,2)), rotation=0, color=col_3)
        plt.text(maxim-1e13, -5.2, "{0}".format(round(maxim/1e14,2)), rotation=0,color=col_2)
        plt.text(maxim-1e13, -7.5, "MAX", rotation=0, color=col_2)

    ax.set_ylabel(f'Counts  /  {bin_width/1e12:.0f} '+'$\\times 10^{12}$ $M_{\odot}$', fontsize=14)
    ax.set_xlabel('Masscut ($M_{\odot}$)', fontsize=14, labelpad=25)
    # ax.set_title('Masscuts Distribution', fontsize=20)
    _, right = plt.xlim()
    ax.set_xlim(0, right)
    ax.legend(fontsize=14, facecolor='white')
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)

    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.xaxis.get_offset_text().set_fontsize(12)
    ax.ticklabel_format(axis="x", style="scientific", scilimits=(0,0))

    plt.savefig("Plots/Masscuts_nwlh.png", bbox_inches='tight', dpi=400)
    plt.close()
    print(f"The linking radius is {hparams.r_link}")
    # number of halos in the dataset
    halos = np.array([graph.x.shape[0] for graph in dataset])
    print(f"Total of halos in the dataset: {halos.sum(0)} \nMean of {halos.mean(0):.1f} halos per simulation \nStd of {halos.std(0):.1f} halos")

    print(f'number of halos in least massive graph: {halos.min()}')

    print('Total of graphs:', len(dataset))
    
    print('Mean of masscut ',np.mean(np.array(masscuts)),'with std ',np.std(np.array(masscuts)))
    
    

    return list(dataset)
