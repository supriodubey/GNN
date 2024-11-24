import optuna
from optuna.storages import RetryFailedTrialCallback  # Import the callback
from optuna.visualization import plot_optimization_history, plot_contour, plot_param_importances
from hyperparameters import hparams
from main import *
import pickle
import os
import time
import datetime

# Function to delete study if exists
def delete_study_if_exists(study_name: str, storage_url: str):
    try:
        # Attempt to load the study
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        
        # If the study is successfully loaded, delete it
        optuna.delete_study(study_name=study_name, storage=storage_url)
        print(f"Study '{study_name}' has been deleted.")

    except KeyError:
        # If the study does not exist, KeyError is raised
        print(f"Study '{study_name}' does not exist, nothing to delete.")

# Objective function to minimize
def objective(trial):
    # Hyperparameters to optimize
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
    T_max =  trial.suggest_int("T_max", 5, 50 , log=False)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    n_layers = trial.suggest_int("n_layers", 1,7)
    hidden_channels = trial.suggest_categorical("hidden_channels", [8, 16, 32, 64])
    dropout_rate = trial.suggest_float("dropout_rate", 1e-2, 0.1, log=True)
    r_link = trial.suggest_float("r_link", 1e-2, 0.2, log=True)
    alpha = trial.suggest_float("alpha", 0.65, 0.95, step=0.05)

    # Print trial information
    print('\nTrial number: {}'.format(trial.number))
    print('\tlearning_rate: {}'.format(learning_rate))
    print('\tT_max: {}'.format(T_max))
    print('\tweight_decay: {}'.format(weight_decay))
    print('\tn_layers:  {}'.format(n_layers))
    print('\thidden_channels:  {}'.format(hidden_channels))
    print('\tdropout_rate: {}'.format(dropout_rate))
    print('\talpha: {}'.format(alpha))
    print('\tr_link:  {}'.format(r_link))
    # Assign hyperparameters to hparams
    hparams.learning_rate = learning_rate
    hparams.T_max = T_max
    hparams.weight_decay = weight_decay
    hparams.n_layers = n_layers
    hparams.hidden_channels = hidden_channels
    hparams.dropout_rate = dropout_rate
    hparams.r_link =r_link
    hparams.alpha = alpha

    # Run main routine
    min_val_loss = main(hparams, verbose=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return min_val_loss

# --- MAIN ---#
if __name__ == "__main__":

    time_ini = time.time()

    # Optuna parameters
    storage = "sqlite:///" + os.path.join("/mnt/", "optuna_QUIJOTE_nwlh.db")
    study_name = "clean_gnn_Mnu"
    total_trials = 30
    
    # Delete study if already present
    print('Deleting study named', study_name)
    delete_study_if_exists(study_name=study_name, storage_url=storage)

    # Define sampler and load/create the study
    sampler = optuna.samplers.TPESampler(n_startup_trials=10)
    
    # Set up the RDBStorage with RetryFailedTrialCallback for automatic retry on failed trials
    storage = optuna.storages.RDBStorage(
        # url="sqlite:///" + os.path.join("/mnt/", "optuna_QUIJOTE.db"),
        url="sqlite:///" + os.path.join("/mnt/", "optuna_QUIJOTE_nwlh.db"),
        heartbeat_interval=60,
        grace_period=120,
        failed_trial_callback=RetryFailedTrialCallback(max_retry=3)  # Retry failed trials up to 3 times
    )
    
    # Create or load the study
    study = optuna.create_study(
        directions=['minimize'],
        study_name=study_name,
        sampler=sampler,
        storage=storage,
        load_if_exists=True
    )
    
    study.optimize(objective, total_trials, gc_after_trial=True)

    # Print info for best trial
    trial = min(study.best_trials, key=lambda t: t.values[0])
    print("Best trial:")
    print("  Validation loss Value: ", trial.values[0])
    # print("  Chi2 Value: ", trial.values[1])
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save best hyperparameters
    best_hpars_file = 'best_hparams_all_r_link_norm.pkl'
    with open(best_hpars_file, 'wb') as file:
        pickle.dump(hparams, file)

    # Visualization of optimization results
    os.makedirs("/mnt/Plots", exist_ok=True)
    fig = plot_optimization_history(study, target=lambda t: t.values[0])
    fig.write_image("/mnt/Plots/optuna_optimization_history.png", width=1200, height=800)
    
    fig = plot_contour(study, target=lambda t: t.values[0])
    fig.write_image("/mnt/Plots/optuna_contour.png", width=1200, height=800)

    fig = plot_param_importances(study, target=lambda t: t.values[0])
    fig.write_image("/mnt/Plots/optuna_param_importances.png", width=1200, height=800)

    print("END --- time elapsed:", datetime.timedelta(seconds=time.time()-time_ini))