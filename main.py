
import time, datetime, psutil

from Source.metalayer import *
from Source.plotting import *
from Source.training import *
from Source.load_data import *

import warnings
import pickle
import os

# global seed function for reproducibility 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.banchmark = False

# Main routine to train the neural net
def main(hparams,testing = False, verbose = True):

    # Ignore unnecessary warnings
    warnings.filterwarnings("ignore")

    # Load data and create dataset
    if verbose: print('\n--- Creating dataset ---\n')
    time_dataset = time.time()
    dataset = create_dataset(hparams)
    print("Dataset created. Time elapsed:", datetime.timedelta(seconds=time.time()-time_dataset))
    node_features = dataset[0].x.shape[1] # 0 if only position here we are taking mass as the node features
    num_params = dataset[0].y.shape[1] 
    # print(f"node features is : {node_features}")

    # Split dataset among training, validation and testing datasets
    train_loader, valid_loader, test_loader = split_datasets(dataset)

    # Size of the output of the GNN 
    dim_out = 2 * num_params # in our case is 2 as mean and variance

    # Initialize model
    model = GNN(node_features=node_features,
                n_layers=hparams.n_layers,
                hidden_channels=hparams.hidden_channels,
                linkradius=hparams.r_link,
                dim_out=dim_out,
                dropout_rate=hparams.dropout_rate,
                alpha=hparams.alpha)
    
    model.to(device)

    # Print the memory (in GB) being used now:
    process = psutil.Process()
    print(f"Memory being used (GB): {process.memory_info().rss/1.e9:.3f}")

    # Train the net
    if verbose: print("\n--- Training ---\n")
    train_losses, valid_losses = training_routine(model,
                                                         train_loader,
                                                         valid_loader,
                                                         test_loader,
                                                         hparams,
                                                         num_params,
                                                         verbose
                                                        )

    # Saving train and validation losses
    current = datetime.datetime.now()
    day = current.strftime("%d")
    current_time = current.strftime("%H_%M")
    os.makedirs('/mnt/Outputs/Losses', exist_ok=True)
    np.save("/mnt/Outputs/Losses/train_loss_"+day+"_"+current_time, train_losses)
    np.save("/mnt/Outputs/Losses/val_loss_"+day+"_"+current_time, valid_losses)

    # Test the model
    # if verbose:
    #     print("\n---- Testing ----\n")
    #     test_loss, rel_err = test(test_loader, model, num_params)
    #     if verbose: print("Test Loss: {:.6f}, Relative error: {:.6f}".format(test_loss, rel_err))
    # else :
    #     print('\n--- Validation mode on ---\n')
    #     print("Validation Loss: {:.6f}, Relative error: {:.6f}".format(valid_losses,rel_err))

    # Plot loss trends
    # plot_losses(train_losses, valid_losses, hparams, display = False)

    # Plot true vs predicted params
    # cosmological_params = ["Om"]#, "f_NL",, "h", "ns", "sig_8"]
    # for param in cosmological_params:
    #     plot_out_true_scatter(hparams, param, display= False)  # Set display=False if you don't need to show the plot
        

    return min(valid_losses) # why where they taking min validation loss?

# --- MAIN ---#

if __name__ == "__main__":
    
    time_ini = time.time()

    set_seed(0)

    # Load hyperparameters
    fname = "best_hparams_all.pkl"
    with open(fname, 'rb') as file:
        best_hparams = pickle.load(file)

    # Changing training epochs
    # best_hparams.n_epochs = 1000

    # print hparams
    print('\nHyperparameters:')
    print('\tlearnig_rate: {}'.format(best_hparams.learning_rate))
    print('\tT_max: {}'.format(best_hparams.T_max))
    print('\tweight_decay: {}'.format(best_hparams.weight_decay))
    print('\tn_layers: {}'.format(best_hparams.n_layers))
    print('\thidden_channels: {}'.format(best_hparams.hidden_channels))
    print('\tnumber of epochs: {}'.format(best_hparams.n_epochs))
    print('\tdropout_rate: {}'.format(best_hparams.dropout_rate))
    print('\tlinking_radius: {}'.format(best_hparams.r_link))
    # print('\talpha: {}'.format(best_hparams.alpha))

    main(best_hparams)

    print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))
