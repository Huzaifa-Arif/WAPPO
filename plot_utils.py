import matplotlib.pyplot as plt
import os
import numpy as np
from variables import variables
import os
import pickle
import torch



def plotloss_curve(losses, max_iters, output_path):
    # Ensure the directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Create a figure and axis
    plt.figure(figsize=(8, 6))

    iterations = list(range(1, max_iters + 1))
    
    # Plot the loss curve
    plt.plot(iterations, losses, label="Loss", color='blue')
    
    # Adding labels and title
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    
    # Adding grid for better readability
    plt.grid(True)
    
    # Append the filename to the output directory
    full_output_path = os.path.join(output_path, "loss_curve.png")
    
    # Save the plot to the specified output path
    plt.savefig(full_output_path)
    
    # Show the plot
    plt.show()

def denormalize(data,params):

    img_shape_x = 720

    # means and stds over training data
    means = params.means
    stds = params.stds
    data = data.cpu().numpy()
    data = data * stds  + means
    return data
 
def visualize_chan(data,params, channel_vis=2,norm = True):
    # The data has shape (T, C, Long, Lat)
    
    time_vis = 0  # Time index to visualize
    field = channel_vis  # Select the channel to visualize

    if(norm == True):
        data = denormalize(data,params)
    else:
        data = data.cpu().numpy()

    global_temp = data[time_vis][field]
    cmap_t = "viridis"  # Colormap choice

    #import pdb;pdb.set_trace()

    # Plot the selected temperature data without subplots
    plt.figure(figsize=(20, 5))
    im0 = plt.imshow(global_temp, cmap=cmap_t)

    # Set plot title
    plt.title("Temperature Ranges")

    # Add colorbar to the plot
    plt.colorbar(im0)

    # Adjust layout to avoid overlapping elements
    plt.tight_layout()

    # Create output directory if it doesn't exist
    output_path = os.path.join("channel_visualization")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Save the figure
    plt.savefig(os.path.join(output_path, "temperature_raw.png"))


def load_tensors_from_pickle(pickle_file_path):
    with open(pickle_file_path, 'rb') as f:
        tensors = pickle.load(f)
    return tensors



def save_rmse_acc_comparison_md(acc_p_cpu, rmse_p_cpu, acc_cpu, rmse_cpu, file_path='results/rmse_acc_comparison.md'):
    """
    Save a comparison table of RMSE and ACC between perturbed and unperturbed inference as a markdown file.

    Parameters:
    - acc_p_cpu: List or numpy array of accuracy values for perturbed inference
    - rmse_p_cpu: List or numpy array of RMSE values for perturbed inference
    - acc_cpu: List or numpy array of accuracy values for unperturbed inference
    - rmse_cpu: List or numpy array of RMSE values for unperturbed inference
    - file_path: File path for saving the markdown file (default is 'rmse_acc_comparison.md')
    """
    timesteps = range(len(acc_p_cpu))

    # Create a markdown table header
    md_table = "| **Timestep** | **t2m RMS Error ($\\delta = 112.53$)** | **ACC ($\\delta = 112.53$)** | **t2m RMS Error ($\\delta = 0$)** | **ACC ($\\delta = 0$)** |\n"
    md_table += "|--------------|------------------------------------|------------------------|--------------------------------|--------------------|\n"

    # Fill the table with data
    for i, t in enumerate(timesteps):
        # Ensure we extract a scalar value if the array is size-1 or handle multidimensional arrays accordingly
        rmse_p_value = float(rmse_p_cpu[i].item()) if np.isscalar(rmse_p_cpu[i]) or rmse_p_cpu[i].size == 1 else float(rmse_p_cpu[i].mean())
        acc_p_value = float(acc_p_cpu[i].item()) if np.isscalar(acc_p_cpu[i]) or acc_p_cpu[i].size == 1 else float(acc_p_cpu[i].mean())
        rmse_value = float(rmse_cpu[i].item()) if np.isscalar(rmse_cpu[i]) or rmse_cpu[i].size == 1 else float(rmse_cpu[i].mean())
        acc_value = float(acc_cpu[i].item()) if np.isscalar(acc_cpu[i]) or acc_cpu[i].size == 1 else float(acc_cpu[i].mean())

        md_table += f"| {t} | {rmse_p_value:.4f} | {acc_p_value:.4f} | {rmse_value:.4f} | {acc_value:.4f} |\n"

    # Save the markdown table to a file
    with open(file_path, 'w') as file:
        file.write(md_table)

    print(f"Markdown table saved to {file_path}")

def plot_results(params,predictions_cpu, targets_cpu, perturbed_predictions_cpu, t_adv, acc_cpu, rmse_cpu, acc_p_cpu, rmse_p_cpu, args):
    
    # Convert tensors to numpy if needed
    scaled = True
    #import pdb;pdb.set_trace()
    t_adv_np = t_adv.cpu().numpy() if isinstance(t_adv, torch.Tensor) else t_adv
    predictions_np = predictions_cpu if isinstance(predictions_cpu, np.ndarray) else predictions_cpu.cpu().numpy()
    perturbed_predictions_np = perturbed_predictions_cpu if isinstance(perturbed_predictions_cpu, np.ndarray) else perturbed_predictions_cpu.cpu().numpy()
    targets_np = targets_cpu if isinstance(targets_cpu, np.ndarray) else targets_cpu.cpu().numpy()
    

    t = args.prediction_length - 1  # at tx6 hours lead time
    cmap_t = "viridis"  # or "bwr"s
    idx_vis = variables.index(args.field)

    if(scaled):
        t_adv_np = (t_adv_np * params.stds) + params.means   
        perturbed_predictions_np[t, 0] = (perturbed_predictions_np[t, 0] * params.stds[idx_vis]) + params.means[idx_vis]
        predictions_np[t, 0] = (predictions_np[t, 0] * params.stds[idx_vis]) + params.means[idx_vis]
        #import pdb;pdb.set_trace()
        targets_np[t,0] = (targets_np[t, 0] * params.stds[idx_vis]) + params.means[idx_vis]

    
   

    ### First Plot: Target Adversary vs Ground Truth vs Perturbed Prediction ###
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))


    im0 = ax[0].imshow(t_adv_np[0, idx_vis], cmap=cmap_t)

    ## Add the case if scaled 
    im1 = ax[1].imshow(targets_np[t, 0], cmap=cmap_t)
    im2 = ax[2].imshow(perturbed_predictions_np[t, 0], cmap=cmap_t)

    ax[0].set_title("Target Adversary")
    ax[1].set_title("ERA5 Ground Truth")
    ax[2].set_title("FourCastNet Perturbed Prediction")

    # Add colorbars to each subplot
    fig.colorbar(im0, ax=ax[0])
    fig.colorbar(im0, ax=ax[1])
    fig.colorbar(im0, ax=ax[2])

    fig.tight_layout()

    #args.output_path = "./scaled_results_mse_sparse"

    output_path = args.output_path  # "./results"

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    fig.savefig(os.path.join(output_path, f"adversary_vs_ground_truth_vs_perturbed_t{t}.png"))

    ### Second Plot: Target Adversary vs Ground Truth vs Unperturbed Prediction ###
    fig2, ax2 = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))

    # Plot t_adv and perturbed predictions with colorbars
    im0 = ax2[0].imshow(t_adv_np[0, idx_vis], cmap=cmap_t)
    im1 = ax2[1].imshow(targets_np[t, 0], cmap=cmap_t)
    im2 = ax2[2].imshow(predictions_np[t, 0], cmap=cmap_t)

    # Set titles
    ax2[0].set_title("Target Adversary")
    ax2[1].set_title("ERA5 Ground Truth")
    ax2[2].set_title("FourCastNet Unperturbed Prediction")

    # Add colorbars to each subplot
    fig2.colorbar(im0, ax=ax2[0])
    fig2.colorbar(im0, ax=ax2[1])
    fig2.colorbar(im0, ax=ax2[2])

    # Adjust layout
    fig2.tight_layout()

    # Save the figure
    fig2.savefig(os.path.join(output_path, f"t_adv_vs_ground_truth_vs_unperturbed_t{t}.png"))

    ### Third Plot: Pointwise Differences and Error Metrics ###
    fig3, ax3 = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    # Compute pointwise differences
    diff_pred_adv = perturbed_predictions_np[t, 0] - t_adv_np[0, idx_vis]
    diff_pred_true = perturbed_predictions_np[t, 0] - targets_np[t, 0]

    # Plot the pointwise differences
    im0 = ax3[0].imshow(diff_pred_adv, cmap='coolwarm')
    im1 = ax3[1].imshow(diff_pred_true, cmap='coolwarm')

    ax3[0].set_title("Difference: Prediction - Adversarial Target")
    ax3[1].set_title("Difference: Prediction - True Value")

    # Add colorbars to each subplot
    fig3.colorbar(im0, ax=ax3[0])
    fig3.colorbar(im0, ax=ax3[1])

    # Adjust layout
    fig3.tight_layout()

    # Save the figure
    fig3.savefig(os.path.join(output_path, f"pointwise_differences_t{t}.png"))

    ### Error Metrics: MSE and Infinity Norm ###
    mse_adv = np.mean(diff_pred_adv ** 2)
    mse_true = np.mean(diff_pred_true ** 2)
    inf_norm_adv = np.max(np.abs(diff_pred_adv))
    inf_norm_true = np.max(np.abs(diff_pred_true))

    print(f"MSE (Prediction vs Adversarial Target): {mse_adv:.4f}")
    print(f"MSE (Prediction vs True Value): {mse_true:.4f}")
    print(f"Infinity Norm (Prediction vs Adversarial Target): {inf_norm_adv:.4f}")
    print(f"Infinity Norm (Prediction vs True Value): {inf_norm_true:.4f}")

    

def plot_trajectory_delta(args, trajectory):
    
    trajectory_length = len(trajectory)
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 5))
    cmap_t = "viridis" # "bwr"
    idx_vis = variables.index(args.field)
    import pdb;pdb.set_trace()
    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Plot the first 10 steps in the trajectory
    for i in range(10):
        if i < trajectory_length:
            t = list(trajectory.keys())[i]  # Get the iteration step
            axes[i].imshow(trajectory[t][0, idx_vis].cpu().numpy(), cmap=cmap_t)
            axes[i].set_title(f"Iteration {t}")
        axes[i].axis('off')  # Turn off the axis

    # Set a main title
    fig.suptitle("FourCastNet Perturbed Delta Trajectory", fontsize=16)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_path = args.output_path #"./results"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    fig.savefig(os.path.join(output_path, f"delta_trajectory.png"))


def plot_trajectory_all(targets_cpu, perturbed_predictions_cpu, t_adv, args):
    
    prediction_length = args.prediction_length
    idx_vis = 0  # Corrected index for single-channel data
    cmap_t = "viridis"  # or "bwr"

    # Create subplots: 2 columns (ground truth, perturbed) and rows equal to the prediction length
    fig, axes = plt.subplots(nrows=prediction_length, ncols=2, figsize=(15, 5 * prediction_length))

    # Flatten axes for easy iteration
    axes = axes.flatten()

    for t in range(prediction_length):
        # Plot ERA5 ground truth
        im = axes[t * 2].imshow(targets_cpu[t, idx_vis], cmap=cmap_t)
        axes[t * 2].set_title(f"Ground Truth at t={t}")
        axes[t * 2].axis('off')
        fig.colorbar(im, ax=axes[t * 2])

        # Plot FourCastNet perturbed prediction
        im = axes[t * 2 + 1].imshow(perturbed_predictions_cpu[t, idx_vis], cmap=cmap_t)
        axes[t * 2 + 1].set_title(f"Perturbed Prediction at t={t}")
        axes[t * 2 + 1].axis('off')
        fig.colorbar(im, ax=axes[t * 2 + 1])

    # Set a main title
    fig.suptitle("Ground Truth vs Perturbed Predictions", fontsize=16)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_path = args.output_path  # "./results"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    fig.savefig(os.path.join(output_path, f"ground_truth_vs_perturbed_full_trajectory.png"))




def save_tensors_to_pickle(delta,acc_p_cpu, rmse_p_cpu, perturbed_predictions_cpu, targets_cpu, trajectory, args):
    # Prepare a dictionary to save in the pickle file
    results = {
        "acc_p_cpu": acc_p_cpu,
        "rmse_p_cpu": rmse_p_cpu,
        "perturbed_predictions_cpu": perturbed_predictions_cpu,
        "targets_cpu": targets_cpu,
        "trajectory": trajectory,
        "delta": delta
    }
    #args.output_path = ""
    # Define the output path and create the directory if it doesn't exist
    output_path = os.path.join(args.output_path, "saved_tensors_mse")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Save the dictionary to a pickle file
    pickle_file_path = os.path.join(output_path, "tensors.pkl")
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"Tensors saved successfully to {pickle_file_path}")

