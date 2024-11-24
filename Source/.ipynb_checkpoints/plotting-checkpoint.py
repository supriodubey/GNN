import matplotlib.pyplot as plt
from Source.constants import *
from sklearn.metrics import r2_score
from matplotlib.offsetbox import AnchoredText
import numpy as np

# Colors for plotting
col_1 = '#648FFF'
col_2 = '#785EF0'
col_3 = '#DC267F'
col_4 = '#FE6100'
col_5 = '#FFB000'

# Plot loss trends
def plot_losses(train_losses, valid_losses, hparams, display):
    
    epochs = len(train_losses)

    fig_losses, ax = plt.subplots(figsize=(8,6))
    ax.grid(alpha=0.4)

    ax.plot(range(epochs), train_losses, label='Training', linewidth=0.8, color=col_1, alpha=0.8)
    ax.plot(range(epochs), valid_losses, label='Validation', linewidth=0.8, color=col_4, alpha=0.8)

    ax.legend()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')

    if display: plt.show()
    fig_losses.savefig("/mnt/Plots/losses.png", bbox_inches='tight', dpi=400)
    plt.close(fig_losses)

# Remove normalization of cosmo parameters
def denormalize(trues, outputs, errors, minpar, maxpar):
    trues = minpar + trues * (maxpar - minpar)
    outputs = minpar + outputs * (maxpar - minpar)
    errors = errors * (maxpar - minpar)
    return trues, outputs, errors

# def std_denormalize(trues, outputs, errors, mean_params, std_params):
#     trues = trues * std_params + mean_params
#     outputs = outputs * std_params + mean_params
#     errors = errors * std_params  # Only scale by std as errors are not shifted by the mean
#     return trues, outputs, errors

# Scatter plot of true vs predicted cosmological parameter
def plot_out_true_scatter(hparams, cosmoparam, display):
    figscat, axscat = plt.subplots(figsize=(8,6))
    axscat.grid(alpha=0.2)

    # Load true values and predicted means and standard deviations
    trues = np.load("/mnt/Outputs/true_values.npy")
    outputs = np.load("/mnt/Outputs/predicted_values.npy")
    errors = np.load("/mnt/Outputs/errors_predicted.npy")

    # Fix initial zero point
    outputs, trues, errors = outputs[1:], trues[1:], errors[1:]
   

    # Define parameter ranges
    param_ranges = {
        # "f_NL": (-600, 600),
        "Om": (0.1, 0.5),
        "Ob":(0.03, 0.07),
        "h": (0.5, 0.9),
        "ns": (0.8, 1.2),
        "sig_8": (0.6, 1.0),
        "M_nu": (0.01,1.0),
    }

    if cosmoparam in param_ranges:
        minpar, maxpar = param_ranges[cosmoparam]
        # print(minpar)
        # idx = list(param_ranges.keys()).index(cosmoparam)
        idx = ["M_nu"].index(cosmoparam)#"f_NL","Om", "h", "ns", "sig_8","Om", "h", "ns", "sig_8"].index(cosmoparam)#["Om"]]["Om","Ob", "h", "ns", "sig_8",["Om"]
        # print(idx)
       
        outputs, trues, errors = outputs[:, idx], trues[:, idx], errors[:, idx]
        

       # print(f"inside plotting the output is :{outputs} \n trues: {trues} \n errors: {errors}" )
    else:
        print("Invalid cosmological parameter")
        return

    trues, outputs, errors = denormalize(trues, outputs, errors, minpar, maxpar)

    # Calculate statistics
    cond_success_1sig = np.abs(outputs - trues) <= np.abs(errors)
    cond_success_2sig = np.abs(outputs - trues) <= 2. * np.abs(errors)
    tot_points = outputs.shape[0]
    successes1sig = outputs[cond_success_1sig].shape[0]
    successes2sig = outputs[cond_success_2sig].shape[0]

    r2 = r2_score(trues, outputs)
    err_rel = np.mean(np.abs((trues - outputs)/(trues)), axis=0)
    chi2 = np.mean((outputs - trues)**2 / errors**2)
    
    print("R^2={:.3f}, Relative error={:.3e}, Chi2={:.3f}".format(r2, err_rel, chi2))
    print(f'Accuracy at 1 sigma: {successes1sig/tot_points:.3f}, at 2 sigma: {successes2sig/tot_points:.3f}')

    # Sort by true value
    indsort = trues.argsort()
    outputs, trues, errors = outputs[indsort], trues[indsort], errors[indsort]

    # Plot predictions vs true values
    truemin, truemax = trues.min(), trues.max()
    axscat.plot([truemin, truemax], [truemin, truemax], color=col_3)
    axscat.errorbar(trues, outputs, yerr=np.abs(errors), marker="o", ls="none", markersize=0.5, elinewidth=0.5, color=col_2)

    # Add legend with metrics
    param_label = {
        # "f_NL": r"$f_{\rm NL}$",
        "Om": r"$\Omega_m$",
        "Ob": r"$\Omega_b$",
        "h": r"$h$",
        "ns": r"$n_s$",
        "sig_8": r"$\sigma_8$",
        "M_nu": r"$M_{\nu}$"
    }
    leg = f"{param_label[cosmoparam]}\n $R^2$={r2:.3f}\n $\epsilon$={100*err_rel:.1f} %\n$ \chi^2$={chi2:.2f}\n" \
          f"Accuracy at $1\sigma$: {100*successes1sig/tot_points:.1f}%" #\n Accuracy at $2\sigma$: {100*successes2sig/tot_points:.1f}%
    at = AnchoredText(leg, frameon=True, loc="upper left", prop=dict(size=12))
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    axscat.add_artist(at)

    axscat.set_ylabel("Prediction", fontsize=14)
    axscat.set_xlabel("Truth", fontsize=14)
    axscat.grid(alpha=0.8)

    if display:
        plt.show()
        figscat.savefig(f"/mnt/Plots/true_vs_pred_{cosmoparam}.png", bbox_inches='tight', dpi=200)
        plt.close(figscat)
