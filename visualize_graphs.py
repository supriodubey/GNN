# Script to visualize halo catalogues as graphs

import time, datetime, os
from Source.plotting import *
from Source.load_data import *
from cluster_radius import radius_graph
import readfof
from tqdm import tqdm
from matplotlib.ticker import ScalarFormatter
from joblib import Parallel, delayed


fontsize = 12

# colors
col_1 = '#648FFF'
col_2 = '#785EF0'
col_3 = '#DC267F'
col_4 = '#FE6100'
col_5 = '#FFB000'


# Visualization routine for plotting graphs
def visualize_graph(num, data, masses, projection="3d", edge_index=None):

    fig = plt.figure(figsize=(12, 12))

    if projection=="3d":
        ax = fig.add_subplot(projection ="3d")
        pos = data.x[:,:3]
    elif projection=="2d":
        ax = fig.add_subplot()
        pos = data.x[:,:2]

    pos *= boxsize/1.e3   # show in Mpc

    # Draw lines for each edge
    if edge_index is not None:
        for (src, dst) in edge_index.t().tolist():

            src = pos[src].tolist()
            dst = pos[dst].tolist()

            if projection=="3d":
                ax.plot([src[0], dst[0]], [src[1], dst[1]], zs=[src[2], dst[2]], linewidth=0.6, color='dimgrey')
            elif projection=="2d":
                ax.plot([src[0], dst[0]], [src[1], dst[1]], linewidth=0.1, color='black')

    # Plot nodes
    if projection=="3d":
        mass_mean = np.mean(masses)
        for i,m in enumerate(masses):
            ax.scatter(pos[i, 0], pos[i, 1], pos[i, 2], s=50*m*m/(mass_mean**2), zorder=1000, alpha=0.6, color = 'mediumpurple')
    elif projection=="2d":
        ax.scatter(pos[:, 0], pos[:, 1], s=m, zorder=1000, alpha=0.5)

    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
    ax.zaxis.set_tick_params(labelsize=fontsize)

    ax.set_xlabel('x (Mpc)', fontsize=16, labelpad=15)
    ax.set_ylabel('y (Mpc)', fontsize=16, labelpad=15)
    ax.set_zlabel('z (Mpc)', fontsize=16, labelpad=15)

    param_file = "/mnt/Halos/latin_hypercube_EQ/latin_hypercube_params.txt" 
    paramsfile = np.loadtxt(param_file, dtype=str)

    rl = '$R_{link} = 0.2$'

    ax.set_title(f'\tGraph n°{num}, Masses $\\geq 99.6$% percentile, {rl} Mpc \t \n \n $\\Omega_m = {float(paramsfile[int(num), 0]):.3f}$ \t $\\sigma_8 = {float(paramsfile[int(num), 1]):.3f}$', fontsize=20)

    fig.savefig("/mnt/Plots/Graphs/graph_"+num+"_996.png", bbox_inches='tight', pad_inches=0.6, dpi=400)
    # plt.close(fig)



# Main routine to display graphs from several simulations
def single_graph_display(simnumber, r_link):
    simpath = f"/mnt/Halos/latin_hypercube_EQ/LH_{simnumber}"
    FoF = readfof.FoF_catalog(simpath, 45,
                              long_ids=False,
                              swap=False,
                              SFR=False,
                              read_IDs=False)
    pos  = FoF.GroupPos / 1e6            # Halo positions in Mpc/h
    mass_raw = FoF.GroupMass * 1e10          # Halo masses in Msun/h

    # cut_val = 3.5e14    # universal mass cut
    cut_val = np.quantile(mass_raw, 0.997)    # NOT universal mass cut
    mass_mask = (mass_raw >= cut_val)
    mass_mask = mass_mask.reshape(-1)
    mass=mass_raw[mass_mask]
    pos=pos[mass_mask]
    
    tab = np.column_stack((pos,mass))

    # edge_index, edge_attr = get_edges(pos, r_link, use_loops=False)
    edge_index = radius_graph(torch.tensor(pos,dtype=torch.float32), r=r_link, loop=False)

    data = Data(x=tab, edge_index=torch.tensor(edge_index, dtype=torch.long))
    visualize_graph(str(simnumber), data, mass, projection="3d", edge_index=data.edge_index)

    # if showgraph:
    #     # visualize_graph(data, simnumber, "2d", edge_index)
        



def display_graphs(n_sims, r_link, njobs, showgraph=True):
    with tqdm(total=n_sims, desc="Processing Graphs") as pbar:
        Parallel(n_jobs=njobs)(delayed(single_graph_display)(simnumber, r_link) for simnumber in range(n_sims))
        pbar.update(n_sims)  # Update the progress bar after completion of all tasks
    


# Show mass distribution for each simulation
def mass_distr_single(n_sims):
    simpath = f"/mnt/Halos/latin_hypercube_EQ/LH_{simnumber}"
    FoF = readfof.FoF_catalog(simpath, 45, long_ids=False,
                      swap=False, SFR=False, read_IDs=False)
    mass_raw = FoF.GroupMass * 1e10          # Halo masses in Msun/h

    param_file = "/mnt/Halos/latin_hypercube_EQ/latin_hypercube_params.txt" 
    paramsfile = np.loadtxt(param_file, dtype=str)

    mass_max = np.max(mass_raw)
    binwidth = 1e13
    bins = np.arange(0, mass_max + binwidth, binwidth)

    nhalos = mass_raw.shape[0]

    fig, ax = plt.subplots(figsize=(10,6))
    ax.grid(alpha=0.4, linestyle='--')

    cut_val97 = np.quantile(mass_raw,0.997)
    cut_val98 = np.quantile(mass_raw,0.998)
    cut_val99 = np.quantile(mass_raw,0.999)

    ax.hist(mass_raw, bins=bins, color = col_2, alpha=0.65, label = 'Mass distribution', zorder=100)
    ax.axvline(cut_val97, ymin=0, ymax=20000, color = col_3, linewidth=1.8, label='99.7% percentile', zorder=200, linestyle = '-')
    ax.axvline(cut_val98, ymin=0, ymax=20000, color = col_4, linewidth=1.8, label='99.8% percentile', zorder=200, linestyle = '--')
    ax.axvline(cut_val99, ymin=0, ymax=20000, color = col_5, linewidth=1.8, label='99.9% percentile', zorder=200, linestyle = '-.')

    ax.set_xlabel('Halo mass ($M_{\\odot}$)', fontsize=14)
    modot = '$M_{\\odot}$'
    ax.set_ylabel('Counts  /  $10^{13}$ '+modot, fontsize=14)

    ax.legend(fontsize=14, loc='upper right', edgecolor='black', title=f'Graph n°{simnumber}', title_fontsize=15)

    nh = f'{nhalos} halos \n $\\Omega_m = {float(paramsfile[int(simnumber), 0]):.3f}$ \n $\\sigma_8 = {float(paramsfile[int(simnumber), 1]):.3f}$'

    ax.text(0.75, 0.485, nh, transform=ax.transAxes, fontsize=14, bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    ax.set_yscale('log')
    _, right = plt.xlim()
    ax.set_xlim(0, right)

    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)

    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.xaxis.get_offset_text().set_fontsize(12)
    ax.ticklabel_format(axis="x", style="scientific", scilimits=(0,0))

    fig.savefig('/mnt/Plots/MassDistr/mass_distr_'+str(simnumber)+'.png', bbox_inches='tight', dpi=400)


def mass_distr(n_sims, n_jobs):
    with tqdm(total=n_sims, desc="Processing Mass Distributions") as pbar:
        Parallel(n_jobs=n_jobs)(delayed(mass_distr_single)(simnumber) for simnumber in range(n_sims))
        pbar.update(n_sims)       

# --- MAIN ---#

if __name__=="__main__":

    time_ini = time.time()

    # Ensure required directories exist
    paths = ["/mnt/Plots/Graphs", "/mnt/Plots/MassDistr"]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
        else:
            print(f"Directory already exists: {path}")

    # Linking radius
    r_link = 0.2
    n_sims = 2
    n_jobs = 20

    # Uncomment to display graphs
    display_graphs(n_sims, r_link,n_jobs)

    # Display mass distribution
    # mass_distr(n_sims)
    

    print("Finished. Time elapsed:", datetime.timedelta(seconds=time.time() - time_ini))