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
import lpips
from sklearn.decomposition import PCA



def reduce_channels(tensor, n_components=3):
    # Assuming data is in shape (batch_size, channels, height, width)
    batch_size, channels, height, width = tensor.size()
    tensor_reshaped = tensor.view(batch_size, channels, -1).permute(0, 2, 1).cpu().detach().numpy()  # Reshape to (batch_size, height*width, channels)
    tensor_reduced = []
    for i in range(batch_size):
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(tensor_reshaped[i])  # Apply PCA (no need for .cpu().numpy())
        tensor_reduced.append(torch.tensor(reduced).permute(1, 0).view(n_components, height, width))
    return torch.stack(tensor_reduced).to(tensor.device)




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



def normalize_tensor(tensor):
    # Normalize a tensor to have values in the range [-1, 1]
    return tensor * 2.0 - 1.0

def adversarial_perturbation_final_target(data_slice, model, t_adv, prediction_length, epsilon, max_iters=10, lr=0.01, loss_func='lpips'):
    # Initialize perturbation
    delta = torch.zeros_like(data_slice, requires_grad=True).to(data_slice.device)
    
    optimizer = optim.Adam([delta], lr=lr)
    losses = []

    loss_func = 'mse'

    # Initialize the LPIPS model if selected
    if loss_func == 'lpips':
        loss_fn = lpips.LPIPS(net='alex').to(data_slice.device)

    for _ in range(max_iters):
        perturbed_data = data_slice + delta

        if(loss_func == 'lpips'):
            #perturbed_data = normalize_tensor(perturbed_data)
            #t_adv = normalize_tensor(t_adv)
        
            perturbed_data = torch.clamp(perturbed_data, -1, 1)  # Ensure valid data range
        
            # Normalize perturbed data and target for LPIPS
           

        # Ensure perturbed data is differentiable
        perturbed_data.requires_grad_()

        sample_length = 1

        # Forward pass with sampling
        sampled_outputs = []
        for _ in range(sample_length):  # Number of samples
            future_pred = model(perturbed_data[0:1])
            for i in range(1, prediction_length):
                future_pred = model(future_pred)
            sampled_outputs.append(future_pred)

        sampled_outputs = torch.stack(sampled_outputs)

        # Compute loss based on selected function
        if loss_func == 'lpips':
            method = 'spec-chan' # channel
            if(method =='avg'):
                sampled_mean = sampled_outputs.mean(dim=0)
                sampled_mean = sampled_mean.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
                t_adv = t_adv.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            elif(method == 'spec-chan'):
                sampled_mean = sampled_outputs.mean(dim=0)
                sampled_mean = sampled_mean[:, :3, :, :]  # Take the first three channels
                t_adv = t_adv[:, :3, :, :]
            elif(method =='PCA'):
                sampled_mean = reduce_channels(sampled_outputs.mean(dim=0))
                t_adv = reduce_channels(t_adv)
            #import pdb; pdb.set_trace()
            loss = loss_fn(sampled_mean, t_adv)
            #loss = loss_fn(sampled_outputs.mean(dim=0), t_adv)
        else:
            loss = torch.nn.functional.mse_loss(sampled_outputs.mean(dim=0), t_adv)
        
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Optional: Apply additional constraints on delta, like epsilon
        # with torch.no_grad():
        #     delta.data = torch.clamp(delta.data, -epsilon, epsilon)

    return delta.detach()



# def adversarial_perturbation_final_target(data_slice, model, t_adv, prediction_length, epsilon, max_iters=10, lr=0.01):
#     # Initialize perturbation
#     delta = torch.zeros_like(data_slice, requires_grad=True).to(data_slice.device)

#     optimizer = optim.Adam([delta], lr=lr)
#     losses = []

#     for _ in range(max_iters):
#         perturbed_data = data_slice + delta
#         perturbed_data = torch.clamp(perturbed_data, 0, 1)  # Ensure valid data range

#         perturbed_data.requires_grad_()
#         sample_length = 1

#         #torch.backends.cudnn.enabled = False

#         # Forward pass with sampling
#         sampled_outputs = []
#         for _ in range(sample_length):  # Number of samples
#             future_pred = model(perturbed_data[0:1])
#             for i in range(1, prediction_length):
#                 future_pred = model(future_pred)
#             sampled_outputs.append(future_pred)

#         sampled_outputs = torch.stack(sampled_outputs)

#         loss_func = 'lpips'


#         # Compute loss to match the final prediction to t_adv
#         if(loss_func == 'lpips'):
#             loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
#             #loss_fn_vgg = lpips.LPIPS(net='vgg') 
#             loss = loss_fn_alex(sampled_outputs.mean(dim=0), t_adv)
            
#         else:
#             loss = torch.nn.functional.mse_loss(sampled_outputs.mean(dim=0), t_adv)


        
#         #print(f"Loss: {loss.item()}")
#         losses.append(loss.item())
#         optimizer.zero_grad()
#         loss.backward()

#         optimizer.step()

#         # plt.figure(figsize=(10, 5))
#         # plt.plot(losses, label='Loss')
#         # plt.xlabel('Iteration')
#         # plt.ylabel('Loss')
#         # plt.title('Loss vs Iterations')
#         # plt.legend()
#         # plt.grid(True)
#         # plt.savefig('loss_vs_iterations.png')
#         # plt.show()


#         #torch.backends.cudnn.enabled = True

#         # Apply constraints
#         #with torch.no_grad():
#         #    delta.data = torch.clamp(delta.data, -epsilon, epsilon)  # Max norm constraint

#     return delta.detach()



# autoregressive inference helper
def inference(data_slice, model, prediction_length, idx):
    # create memory for the different stats
    n_out_channels = params['N_out_channels']
    acc = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    rmse = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)

    # to conserve GPU mem, only save one channel (can be changed if sufficient GPU mem or move to CPU)
    targets = torch.zeros((prediction_length, 1, img_shape_x, img_shape_y)).to(device, dtype=torch.float)
    predictions = torch.zeros((prediction_length, 1, img_shape_x, img_shape_y)).to(device, dtype=torch.float)


    with torch.no_grad():
        for i in range(data_slice.shape[0]):
            if i == 0:
                first = data_slice[0:1]
                future = data_slice[1:2]
                pred = first
                tar = first
                # also save out predictions for visualizing channel index idx
                targets[0,0] = first[0,idx]
                predictions[0,0] = first[0,idx]
                # predict
                future_pred = model(first)
            else:
                if i < prediction_length - 1:
                    future = data_slice[i+1:i+2]
                future_pred = model(future_pred) # autoregressive step

            if i < prediction_length - 1:
                predictions[i+1,0] = future_pred[0,idx]
                targets[i+1,0] = future[0,idx]

            # compute metrics using the ground truth ERA5 data as "true" predictions
            rmse[i] = weighted_rmse_channels(pred, tar) * std
            acc[i] = weighted_acc_channels(pred-m, tar-m)
            import pdb;pdb.set_trace()
            print('Predicted Unperturbed timestep {} of {}. {} RMS Error: {}, ACC: {}'.format(i, prediction_length, field, rmse[i,idx], acc[i,idx]))

            pred = future_pred
            tar = future

    # copy to cpu for plotting/vis
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
    perturbed_data_slice = data_slice 
    delta_norm = torch.norm(delta).item()
    print(f"Norm of delta: {delta_norm}")

    

    with torch.no_grad():
        for i in range(data_slice.shape[0]):
            if i == 0:
                first = perturbed_data_slice[0:1] + delta
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

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# in and out channels: FourCastNet uses 20 input channels corresponding to 20 prognostic variables
in_channels = np.array(params.in_channels)
out_channels = np.array(params.out_channels)
params['N_in_channels'] = len(in_channels)
params['N_out_channels'] = len(out_channels)
params.means = np.load(global_means_path)[0, out_channels] # for normalizing data with precomputed train stats
params.stds = np.load(global_stds_path)[0, out_channels]
params.time_means = np.load(time_means_path)[0, out_channels]

# load the model
if params.nettype == 'afno':
    model = AFNONet(params).to(device)  # AFNO model
else:
    raise Exception("not implemented")
# load saved model weights
model = load_model(model, params, model_path)
model = model.to(device)
import pdb;pdb.set_trace()



# move normalization tensors to gpu
# load time means: represents climatology
img_shape_x = 720
img_shape_y = 1440

# means and stds over training data
means = params.means
stds = params.stds

# load climatological means
time_means = params.time_means # temporal mean (for every pixel)
m = torch.as_tensor((time_means - means)/stds)[:, 0:img_shape_x]
m = torch.unsqueeze(m, 0)
# these are needed to compute ACC and RMSE metrics
m = m.to(device, dtype=torch.float)
std = torch.as_tensor(stds[:,0,0]).to(device, dtype=torch.float)


#print("Shape of time means = {}".format(m.shape))
#print("Shape of std = {}".format(std.shape))


# setup data for inference
dt = 1 # time step (x 6 hours)
ic = 0 # start the inference from here
prediction_length = 4 # 1-4 work for Adversarial Perturbation number of steps (x 6 hours)


# which field to track for visualization
field = 't2m' #'u10'
idx_vis = variables.index(field) # also prints out metrics for this field

# get prediction length slice from the data
print('Loading inference data')
print('Inference data from {}'.format(data_file))
data = h5py.File(data_file, 'r')['fields'][ic:(ic+prediction_length*dt):dt,in_channels,0:img_shape_x]
print(data.shape)
#import sys;
#sys.exit()
print("Shape of data = {}".format(data.shape))



# run inference
data = (data - means)/stds # standardize the data
data = torch.as_tensor(data).to(device, dtype=torch.float) # move to gpu for inference
epsilon = 0.1

# The goal is to make future predictions time step

t_adv = h5py.File(data_file, 'r')['fields'][ic + 19:ic + 20,in_channels,0:img_shape_x]  # loading the 20th time step as t_adv
t_adv = (t_adv - means) / stds  # standardize the adversarial target
t_adv = torch.as_tensor(t_adv).to(device, dtype=torch.float)  # move to GPU for inference
acc_cpu, rmse_cpu, predictions_cpu, targets_cpu = inference(data, model, prediction_length, idx=idx_vis)
acc_p_cpu, rmse_p_cpu, perturbed_predictions_cpu, targets_cpu = perturbed_inference(data, model, prediction_length, idx=idx_vis, t_adv=t_adv, epsilon=epsilon)



##### Ground Truth vs Prediction vs Perturbed ###

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
t = prediction_length - 1  # at tx6 hours lead time
cmap_t = "viridis" # "bwr"
idx_vis = variables.index(field)
#import pdb;pdb.set_trace()
ax[0].imshow(predictions_cpu[t,0], cmap= cmap_t )
ax[1].imshow(targets_cpu[t,0], cmap= cmap_t )
ax[2].imshow(perturbed_predictions_cpu[t,0], cmap= cmap_t )
ax[0].set_title("FourCastNet prediction")
ax[1].set_title("ERA5 ground truth")
ax[2].set_title("FourCastNet Perturbed prediction")
fig.tight_layout()

output_path = "./results_{1}"
if not os.path.exists(output_path):
    os.makedirs(output_path)
fig.savefig(os.path.join(output_path, f"prediction_vs_ground_truth_vs_perturbed_t{t}.png"))


##### Target vs Prediction vs Perturbed ###
# Create the figure and axis objects for t_adv vs perturbed predictions
fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

t_adv = t_adv.cpu().numpy()

# Plot t_adv and perturbed predictions
ax2[0].imshow(t_adv[0, idx_vis], cmap=cmap_t)
ax2[1].imshow(perturbed_predictions_cpu[t, 0], cmap=cmap_t)

# Set titles
ax2[0].set_title("Adversarial target (t_adv)")
ax2[1].set_title("Perturbed prediction")

# Adjust layout
fig2.tight_layout()

# Save the figure
fig2.savefig(os.path.join(output_path, f"t_adv_vs_perturbed_prediction_t{t}.png"))




#### Plot the RMSE and ACC #####

fig3, ax3 = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

start = 0
end = prediction_length  # < prediction_length forecast

field = 't2m'  # Change this to other fields such as z500
idx_metric = variables.index(field)  # Plot metrics for this field

hrs = np.arange(0, end * 6, 6)

# Plot original ACC and RMSE
ax3[0].plot(hrs, acc_cpu[start:end, idx_metric], "o-", label="Original ACC", ms=4, lw=0.7, color="r")
ax3[1].plot(hrs, rmse_cpu[start:end, idx_metric], "o-", label="Original RMSE", ms=4, lw=0.7, color="r")

# Plot perturbed ACC and RMSE
ax3[0].plot(hrs, acc_p_cpu[start:end, idx_metric], "o-", label="Perturbed ACC", ms=4, lw=0.7, color="b")
ax3[1].plot(hrs, rmse_p_cpu[start:end, idx_metric], "o-", label="Perturbed RMSE", ms=4, lw=0.7, color="b")

# Add legends
ax3[0].legend()
ax3[1].legend()

# Formatting
fsz = "15"
xlist = np.arange(0, end * 6 + 24, 24)
ax3[0].set_xlabel("forecast time (in hrs)", fontsize=fsz)
ax3[1].set_xlabel("forecast time (in hrs)", fontsize=fsz)
ax3[0].set_ylabel("ACC", fontsize=fsz)
ax3[1].set_ylabel("RMSE", fontsize=fsz)
ax3[0].set_ylim(0.3, 1.05)
ax3[0].set_xticks(xlist)
ax3[1].set_xticks(xlist)
ax3[0].tick_params(axis='both', which='both', labelsize=12)
ax3[1].tick_params(axis='both', which='both', labelsize=12)
fig3.tight_layout()

# Save the RMSE and ACC comparison plot
fig3.savefig(os.path.join(output_path, f"rmse_acc_comparison_t{t}.png"))





