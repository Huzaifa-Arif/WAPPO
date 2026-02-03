import torch
import os, sys, time
import numpy as np
import h5py
import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from utils.YParams import YParams
from networks.afnonet import AFNONet
from collections import OrderedDict
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()


# define metrics from the definitions above
def lat(j: torch.Tensor, num_lat: int) -> torch.Tensor:
    return 90. - j * 180./float(num_lat-1)

def latitude_weighting_factor(j: torch.Tensor, num_lat: int, s: torch.Tensor) -> torch.Tensor:
    return num_lat * torch.cos(3.1416/180. * lat(j, num_lat))/s

def weighted_rmse_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each channel
    num_lat = pred.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sqrt(torch.mean(weight * (pred - target)**2., dim=(-1,-2)))
    return result

def weighted_acc_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted acc for each channel
    num_lat = pred.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sum(weight * pred * target, dim=(-1,-2)) / torch.sqrt(torch.sum(weight * pred * pred, dim=(-1,-2)) * torch.sum(weight * target *
    target, dim=(-1,-2)))
    return result



def print_hdf5_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}, Shape: {obj.shape}, Type: {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"Group: {name}")



def adversarial_perturbation_sequence(data_slice, model, t_adv, prediction_length, epsilon, max_iters=100, lr=0.01):
    # Initialize perturbation
    delta = torch.zeros_like(data_slice, requires_grad=True).to(data_slice.device)

    optimizer = optim.Adam([delta], lr=lr)

    for _ in range(max_iters):
        perturbed_data = data_slice + delta
        perturbed_data = torch.clamp(perturbed_data, 0, 1)  # Ensure valid data range

        model_output = []
        with torch.no_grad():
            for i in range(prediction_length):
                if i == 0:
                    future_pred = model(perturbed_data[0:1])
                else:
                    future_pred = model(future_pred)
                model_output.append(future_pred)
            model_output = torch.stack(model_output).squeeze()
        
        
        
        loss = torch.nn.functional.mse_loss(model_output, t_adv)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Apply constraints
        with torch.no_grad():
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)  # Max norm constraint

    return delta.detach()

def adversarial_perturbation_final_target(data_slice, model, t_adv, prediction_length, epsilon, max_iters=10, lr=0.01):
    # Initialize perturbation
    delta = torch.zeros_like(data_slice, requires_grad=True).to(data_slice.device)

    optimizer = optim.Adam([delta], lr=lr)
    losses = []

    for _ in range(max_iters):
        perturbed_data = data_slice + delta
        perturbed_data = torch.clamp(perturbed_data, 0, 1)  # Ensure valid data range

        perturbed_data.requires_grad_()
        sample_length = 1

        #torch.backends.cudnn.enabled = False

        # Forward pass with sampling
        sampled_outputs = []
        for _ in range(sample_length):  # Number of samples
            future_pred = model(perturbed_data[0:1])
            for i in range(1, prediction_length):
                future_pred = model(future_pred)
            sampled_outputs.append(future_pred)

        sampled_outputs = torch.stack(sampled_outputs)

        # Compute loss to match the final prediction to t_adv
        loss = torch.nn.functional.mse_loss(sampled_outputs.mean(dim=0), t_adv)
        #print(f"Loss: {loss.item()}")
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # plt.figure(figsize=(10, 5))
        # plt.plot(losses, label='Loss')
        # plt.xlabel('Iteration')
        # plt.ylabel('Loss')
        # plt.title('Loss vs Iterations')
        # plt.legend()
        # plt.grid(True)
        # plt.savefig('loss_vs_iterations.png')
        # plt.show()


        #torch.backends.cudnn.enabled = True

        # Apply constraints
        with torch.no_grad():
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)  # Max norm constraint

    return delta.detach()



def inference(data_slice, model, prediction_length, idx):
    # Convert data to float32 before passing it to the model
    data_slice = torch.as_tensor(data_slice, dtype=torch.float32).to(device)  # Use the device variable

    # Create memory for the different stats
    n_out_channels = params['N_out_channels']
    acc = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    rmse = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)

    # To conserve GPU memory, only save one channel (can be changed if sufficient GPU memory or move to CPU)
    targets = torch.zeros((prediction_length, 1, img_shape_x, img_shape_y)).to(device, dtype=torch.float)
    predictions = torch.zeros((prediction_length, 1, img_shape_x, img_shape_y)).to(device, dtype=torch.float)

    with torch.no_grad():
        for i in range(data_slice.shape[0]):
            if i == 0:
                first = data_slice[0:1]
                future = data_slice[1:2]
                pred = first
                tar = first
                # Also save out predictions for visualizing channel index idx
                targets[0,0] = first[0,idx]
                predictions[0,0] = first[0,idx]
                # Predict
                future_pred = model(first)
            else:
                if i < prediction_length - 1:
                    future = data_slice[i+1:i+2]
                future_pred = model(future_pred)  # Autoregressive step

            if i < prediction_length - 1:
                predictions[i+1,0] = future_pred[0,idx]
                targets[i+1,0] = future[0,idx]

            # Compute metrics using the ground truth ERA5 data as "true" predictions
            rmse[i] = weighted_rmse_channels(pred, tar) * std
            acc[i] = weighted_acc_channels(pred-m, tar-m)
            print('Predicted Unperturbed timestep {} of {}. {} RMS Error: {}, ACC: {}'.format(i, prediction_length, field, rmse[i,idx], acc[i,idx]))

            pred = future_pred
            tar = future

    # Copy to CPU for plotting/visualization
    acc_cpu = acc.cpu().numpy()
    rmse_cpu = rmse.cpu().numpy()
    predictions_cpu = predictions.cpu().numpy()
    targets_cpu = targets.cpu().numpy()

    return acc_cpu, rmse_cpu, predictions_cpu, targets_cpu


def perturbed_inference(data_slice, model, prediction_length, idx, t_adv, epsilon):
    # create memory for the different stats
    n_out_channels = params['N_out_channels']
    acc = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    rmse = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)

    # to conserve GPU mem, only save one channel (can be changed if sufficient GPU mem or move to CPU)
    targets = torch.zeros((prediction_length, 1, img_shape_x, img_shape_y)).to(device, dtype=torch.float)
    predictions = torch.zeros((prediction_length, 1, img_shape_x, img_shape_y)).to(device, dtype=torch.float)

    # Get adversarial perturbation

    # Option 1 : If the target adversary is a sequence
    #delta = adversarial_perturbation_sequence(data_slice, model, t_adv, prediction_length, epsilon)

    # Option 2: If the target adversary is an image 
    delta = adversarial_perturbation_final_target(data_slice, model, t_adv, prediction_length, epsilon)

    # Apply perturbation to the initial data slice
    perturbed_data_slice = data_slice + delta

    with torch.no_grad():
        for i in range(data_slice.shape[0]):
            if i == 0:
                first = perturbed_data_slice[0:1]
                future = perturbed_data_slice[1:2]
                pred = first
                tar = first
                # also save out predictions for visualizing channel index idx
                targets[0,0] = first[0,idx]
                predictions[0,0] = first[0,idx]
                # predict
                future_pred = model(first)
            else:
                if i < prediction_length - 1:
                    future = perturbed_data_slice[i+1:i+2]
                future_pred = model(future_pred) # autoregressive step

            if i < prediction_length - 1:
                predictions[i+1,0] = future_pred[0,idx]
                targets[i+1,0] = future[0,idx]

            # compute metrics using the ground truth ERA5 data as "true" predictions
            rmse[i] = weighted_rmse_channels(pred, tar) * std
            acc[i] = weighted_acc_channels(pred-m, tar-m)
            print('Predicted Perturbed timestep {} of {}. {} RMS Error: {}, ACC: {}'.format(i, prediction_length, field, rmse[i,idx], acc[i,idx]))

            pred = future_pred
            tar = future

    # copy to cpu for plotting/vis
    acc_cpu = acc.cpu().numpy()
    rmse_cpu = rmse.cpu().numpy()
    predictions_cpu = predictions.cpu().numpy()
    targets_cpu = targets.cpu().numpy()

    return acc_cpu, rmse_cpu, predictions_cpu, targets_cpu


# data and model paths
data_path = "./FourCast_Dataset/data/FCN_ERA5_data_v0/out_of_sample"
data_file = os.path.join(data_path, "2018.h5")
model_path = "./FourCast_Dataset/model_weights/FCN_weights_v0/backbone.ckpt"
global_means_path = "./FourCast_Dataset/additional/stats_v0/global_means.npy"
global_stds_path = "./FourCast_Dataset/additional/stats_v0/global_stds.npy"
time_means_path = "./FourCast_Dataset/additional/stats_v0/time_means.npy"
land_sea_mask_path = "./FourCast_Dataset/additional/stats_v0/land_sea_mask.npy"



# We are going to use a default config. Please see github repo for other config examples
config_file = "./config/AFNO.yaml"
config_name = "afno_backbone"
params = YParams(config_file, config_name)


# Open the file in read mode
with h5py.File(data_file, 'r') as f:
    # Use the visititems method to apply the print_hdf5_structure function to each object in the file
    f.visititems(print_hdf5_structure)





'''
The ordering of atmospheric variables along the channel dimension is as follows:
'''
variables = ['u10',
             'v10',
             't2m',
             'sp',
             'msl',
             't850',
             'u1000',
             'v1000',
             'z1000',
             'u850',
             'v850',
             'z850',
             'u500',
             'v500',
             'z500',
             't500',
             'z50' ,
             'r500',
             'r850',
             'tcwv']



#variables = ['t850', 't2m', 't500', 'tcwv'] 



def load_model(model, params, checkpoint_file):
    ''' helper function to load model weights '''
    checkpoint_fname = checkpoint_file
    checkpoint = torch.load(checkpoint_fname)
    try:
        ''' FourCastNet is trained with distributed data parallel
            (DDP) which prepends 'module' to all keys. Non-DDP
            models need to strip this prefix '''
        new_state_dict = OrderedDict()
        for key, val in checkpoint['model_state'].items():
            name = key[7:]
            if name != 'ged':
                new_state_dict[name] = val
        model.load_state_dict(new_state_dict)
    except:
        model.load_state_dict(checkpoint['model_state'])
    model.eval() # set to inference mode
    return model

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

def train_single_gpu(params):
    # Load model and move it to the correct GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AFNONet(params).to(device)
    model = load_model(model, params, model_path)

    # Get adversarial perturbation and run inference
    print('Running on Single GPU')
    acc_cpu, rmse_cpu, predictions_cpu, targets_cpu = inference(data, model, prediction_length, idx=idx_vis)
    print("Inference complete")

    # Save results
    save_results(acc_cpu, rmse_cpu, predictions_cpu, targets_cpu)


def train(rank, world_size, params):
    setup(rank, world_size)

    # Load model and move it to the correct GPU
    model = AFNONet(params).to(rank)
    model = DDP(model, device_ids=[rank])

    # Get adversarial perturbation and run inference
    print('in')
   # acc_p_cpu, rmse_p_cpu, perturbed_predictions_cpu, targets_cpu = perturbed_inference(data, model, prediction_length, idx=idx_vis, t_adv=t_adv, epsilon=epsilon)
    acc_cpu, rmse_cpu, predictions_cpu, targets_cpu = inference(data, model, prediction_length, idx=idx_vis)
    print("out")
    # Only save results if the rank is 0 (to avoid duplicate saving)
    if rank == 0:
        save_results(acc_cpu, rmse_cpu, predictions_cpu, targets_cpu)
        #save_results(acc_p_cpu, rmse_p_cpu, perturbed_predictions_cpu, targets_cpu)

    cleanup()

def save_results(acc_p_cpu, rmse_p_cpu, perturbed_predictions_cpu, targets_cpu):
    ##### Ground Truth vs Prediction vs Perturbed ###
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
    t = prediction_length - 1  # at tx6 hours lead time
    cmap_t = "viridis" # "bwr"
    idx_vis = variables.index(field)
    ax[0].imshow(perturbed_predictions_cpu[t,0], cmap=cmap_t)
    ax[1].imshow(targets_cpu[t,0], cmap=cmap_t)
    ax[2].imshow(perturbed_predictions_cpu[t,0], cmap=cmap_t)
    ax[0].set_title("FourCastNet prediction")
    ax[1].set_title("ERA5 ground truth")
    ax[2].set_title("FourCastNet Perturbed prediction")
    fig.tight_layout()

    output_path = "./results"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    fig.savefig(os.path.join(output_path, f"prediction_vs_ground_truth_vs_perturbed_t{t}.png"))

    ##### Target vs Prediction vs Perturbed ###
    fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    t_adv = t_adv.cpu().numpy()

    ax2[0].imshow(t_adv[0, idx_vis], cmap=cmap_t)
    ax2[1].imshow(perturbed_predictions_cpu[t, 0], cmap=cmap_t)

    ax2[0].set_title("Adversarial target (t_adv)")
    ax2[1].set_title("Perturbed prediction")

    fig2.tight_layout()
    fig2.savefig(os.path.join(output_path, f"t_adv_vs_perturbed_prediction_t{t}.png"))

    #### Plot the RMSE and ACC #####
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    start = 0
    end = prediction_length

    field = 't2m' # change this to other fields such as z500
    idx_metric = variables.index(field)

    hrs = np.arange(0, end*6, 6)
    ax[0].plot(hrs, acc_cpu[start:end, idx_metric], "o-", label="FourCastNet", ms=4, lw=0.7, color="r")
    ax[1].plot(hrs, rmse_cpu[start:end, idx_metric], "o-", label="FourCastNet", ms=4, lw=0.7, color="r")
    ax[0].legend()
    ax[1].legend()
    fsz = "15"
    xlist = np.arange(0, end*6+24, 24)
    ax[0].set_xlabel("forecast time (in hrs)", fontsize=fsz)
    ax[1].set_xlabel("forecast time (in hrs)", fontsize=fsz)
    ax[0].set_ylabel("ACC", fontsize=fsz)
    ax[1].set_ylabel("RMSE", fontsize=fsz)
    ax[0].set_ylim(0.3, 1.05)
    ax[0].set_xticks(xlist)
    ax[1].set_xticks(xlist)
    ax[0].tick_params(axis='both', which='both', labelsize=12)
    ax[1].tick_params(axis='both', which='both', labelsize=12)
    fig.tight_layout()

    # Save the plot
    fig.savefig(os.path.join(output_path, "acc_rmse_plot.png"))


def main():
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29502'

    world_size = torch.cuda.device_count()

    config_file = "./config/AFNO.yaml"
    config_name = "afno_backbone"
    params = YParams(config_file, config_name)

    in_channels = np.array(params.in_channels)
    out_channels = np.array(params.out_channels)
    params['N_in_channels'] = len(in_channels)
    params['N_out_channels'] = len(out_channels)
    params.means = np.load(global_means_path)[0, out_channels]
    params.stds = np.load(global_stds_path)[0, out_channels]
    params.time_means = np.load(time_means_path)[0, out_channels]
    
    train_single_gpu(params)
    #mp.spawn(train, args=(world_size, params), nprocs=world_size, join=True)




if __name__ == "__main__":
    # Setup the environment
    data_path = "./FourCast_Dataset/data/FCN_ERA5_data_v0/out_of_sample"
    data_file = os.path.join(data_path, "2018.h5")
    model_path = "./FourCast_Dataset/model_weights/FCN_weights_v0/backbone.ckpt"
    global_means_path = "./FourCast_Dataset/additional/stats_v0/global_means.npy"
    global_stds_path = "./FourCast_Dataset/additional/stats_v0/global_stds.npy"
    time_means_path = "./FourCast_Dataset/additional/stats_v0/time_means.npy"
    land_sea_mask_path = "./FourCast_Dataset/additional/stats_v0/land_sea_mask.npy"

    config_file = "./config/AFNO.yaml"
    config_name = "afno_backbone"
    params = YParams(config_file, config_name)

    # in and out channels: FourCastNet uses 20 input channels corresponding to 20 prognostic variables
    in_channels = np.array(params.in_channels)
    out_channels = np.array(params.out_channels)
    params['N_in_channels'] = len(in_channels)
    params['N_out_channels'] = len(out_channels)
    params.means = np.load(global_means_path)[0, out_channels]
    params.stds = np.load(global_stds_path)[0, out_channels]
    params.time_means = np.load(time_means_path)[0, out_channels]

    img_shape_x = 720
    img_shape_y = 1440

    # move normalization tensors to GPU
    means = params.means
    stds = params.stds
    time_means = params.time_means
    m = torch.as_tensor((time_means - means)/stds)[:, 0:img_shape_x]
    m = torch.unsqueeze(m, 0)
    std = torch.as_tensor(stds[:,0,0])

    # setup data for inference
    dt = 1
    ic = 0
    prediction_length = 2
    field = 't2m'
    idx_vis = variables.index(field)

    print('Loading inference data')
    data = h5py.File(data_file, 'r')['fields'][ic:(ic+prediction_length*dt):dt, in_channels, 0:img_shape_x]
    data = (data - means) / stds
    data = torch.as_tensor(data)
    epsilon = 0.1
    t_adv = h5py.File(data_file, 'r')['fields'][ic + 19:ic + 20, in_channels, 0:img_shape_x]
    t_adv = (t_adv - means) / stds
    t_adv = torch.as_tensor(t_adv)

    main()






