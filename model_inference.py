import torch
import torch.optim as optim
from eval_metrics import weighted_rmse_channels, weighted_acc_channels
import lpips
from lpips import LPIPS
from adversarial_model_utils import CustomLPIPS,apply_channel_mask,apply_patch_mask
from sklearn.decomposition import PCA
from eval_metrics import compute_image_similarity_metrics
from adversarial_model_utils import MSELossWithFourierPerturbations,total_variation_per_channel
from tqdm import tqdm
from torch.nn.functional import mse_loss
from torchmetrics.functional.image import total_variation
import os
from plot_utils import plotloss_curve
from torch.utils.tensorboard import SummaryWriter

img_shape_x = 720
img_shape_y = 1440

def get_mean_std(params):
    # means and stds over training data
    means = params.means
    stds = params.stds

    # load climatological means
    time_means = params.time_means # temporal mean (for every pixel)
    m = torch.as_tensor((time_means - means)/stds)[:, 0:img_shape_x]
    m = torch.unsqueeze(m, 0)
    # these are needed to compute ACC and RMSE metrics
    m = m.to(params.device, dtype=torch.float)
    std = torch.as_tensor(stds[:,0,0]).to(params.device, dtype=torch.float)
    return m,std

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

def normalize_tensor(tensor):
    # Normalize a tensor to have values in the range [-1, 1]
    return tensor * 2.0 - 1.0



def adversarial_perturbation_final_target(args,params,data_slice, model, t_adv, prediction_length, epsilon, max_iters=1, lr=0.01, loss_func='lpips + mse'):
    
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    writer = SummaryWriter(log_dir=args.log_dir)

    
    # Initialize perturbation

    delta = torch.zeros_like(data_slice[0:1], requires_grad=True).to(data_slice.device)
    optimizer = optim.Adam([delta], lr=lr)
    channel_mask = False
    patch_mask = True
    total_loss = []

    num_channels = data_slice.shape[1]  # Assuming data_slice has shape [B, N, M, L]



    # Initialize the loss function
    if loss_func == 'lpips + mse':
        original_lpips_model = LPIPS(net='vgg')
        custom_lpips_model = CustomLPIPS(original_lpips_model)
        loss_fn = custom_lpips_model.to(data_slice.device)
    elif loss_func == 'Fourier':
        loss_fn = MSELossWithFourierPerturbations()
    else:
        raise NotImplementedError("Loss function not implemented")
    
    # Dictionary to store trajectories
    trajectory_dict = {}
   
    #max_iters = 2
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1)

    for iter_nums in tqdm(range(max_iters), desc="Optimization Iterations", leave=False):

        
        perturbed_data = data_slice[0:1] + delta
        tv_loss_penalty = 0
        linfty_loss_penalty = 0
        
        # Forward pass with the model
        future_pred = model(perturbed_data)
        for i in range(1, prediction_length):

            future_pred = model(future_pred)
            # Output size for both loss : [batch_size, num_channels].
            tv_loss = total_variation_per_channel(future_pred)
            linfty_loss = torch.amax(torch.abs(future_pred), dim=(2, 3))
            

            # Compute the inner product for the TV and L_infty penalties per channel
            #  The epsilon and tau values can be computed offline
            args.tv_weight   = args.tv_weight.to(params.device)
            args.linfty_weight = args.linfty_weight.to(params.device)
            args.tau = args.tau.to(params.device)
            args.epsilon = args.epsilon.to(params.device)
            tv_loss_penalty += torch.dot(args.tv_weight, torch.relu(tv_loss - args.tau))  # TV penalty (inner product)
            #import pdb;pdb.set_trace()
            linfty_loss_penalty += torch.dot(args.linfty_weight, torch.relu(linfty_loss.squeeze(0) - args.epsilon))  # L_infty penalty (inner product)
        
        # Compute the loss
        if loss_func == 'lpips + mse':
            #loss = (args.lamda * loss_fn(future_pred, t_adv) +
            loss = args.gamma * mse_loss(future_pred, t_adv)
        

            ## Here we want to add the perturbation 
        elif loss_func == 'Fourier':
            loss = loss_fn(future_pred, t_adv)

            ## Add constraints in the loss
        else:
            raise NotImplementedError("Unsupported loss function")
        
        

        # Add these penalties to the loss
        #import pdb;pdb.set_trace()
        loss += tv_loss_penalty
        #loss += linfty_loss_penalty

        # Backpropagation and optimization step
        optimizer.zero_grad()
        total_loss.append(loss.item())
        # Log the loss to TensorBoard
        writer.add_scalar("Loss/total_loss", loss.item(), iter_nums)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_([delta], max_norm=0.002)
        
       

        # Q: Does torch.no_grad() affect the perturbation
        # Project delta to ensure it remains channel- and patch-sparse (Projected Gradient Descent)
        with torch.no_grad():
            # Reapply the channel and patch masks after each gradient step
            if channel_mask == True:  
                channel_mask_tensor = torch.zeros(delta.shape).to(delta.device) 
                
                
                # Iterating over each input channels in a list
                for ch in args.channel_perturb :
                    channel_mask_tensor[:, ch, :, :] = 1
             
                # Apply the mask to delta (only applies delta to the specified channel)
                
                temp = delta.grad.clone()  # Create a copy of the original gradient

                ### Potential;  SGD
                delta.grad = delta.grad * channel_mask_tensor  # Use assignment instead of in-place modification
                print(torch.norm(temp))
                print(torch.norm(delta.grad))

                #import pdb;pdb.set_trace()


                #delta = delta * channel_mask_tensor

        #     # Patch mask (apply delta only to a patch in the spatial dimensions)
            if patch_mask == True:
               
                patch_mask_tensor = apply_patch_mask(
                    delta, 
                    args.patch_start_L, 
                    args.patch_start_M, 
                    args.patch_size_L, 
                    args.patch_size_M
                )
                temp = delta.grad.clone()
                delta.grad = delta.grad *  patch_mask_tensor
                print(torch.norm(temp))
                print(torch.norm(delta.grad))


        optimizer.step()
        scheduler.step(loss.item())

        # # Stop the optimization if the loss goes below a certain threshold
        args.loss_threshold = 0.00003
        if loss.item() < args.loss_threshold:
             print(f"Stopping early at iteration {iter_nums} due to loss threshold")
             max_iters = iter_nums + 1
             break
        # Save trajectory every 100 iterations
        if (i + 1) % 100 == 0:
            trajectory_dict[i + 1] = future_pred.detach().clone()
    # Close the TensorBoard writer
    writer.close()

    #import pdb;pdb.set_trace()
    plotloss_curve(total_loss,max_iters,args.output_path)
    return delta.detach() , trajectory_dict

def inference(params,args,data_slice, model, prediction_length, idx,t_adv):
    # create memory for the different stats
    n_out_channels = params['N_out_channels']
    acc = torch.zeros((prediction_length, n_out_channels)).to(params.device, dtype=torch.float)
    rmse = torch.zeros((prediction_length, n_out_channels)).to(params.device, dtype=torch.float)
    
    # to conserve GPU mem, only save one channel (can be changed if sufficient GPU mem or move to CPU)
    targets = torch.zeros((prediction_length, 1, img_shape_x, img_shape_y)).to(params.device, dtype=torch.float)
    predictions = torch.zeros((prediction_length, 1, img_shape_x, img_shape_y)).to( params.device,dtype=torch.float)
    
    similarity_metrics = {metric: torch.zeros((prediction_length, n_out_channels)).to(params.device, dtype=torch.float)
                          for metric in ['psnr','lpips', 'ssim', 'ms_ssim', 'iw_ssim', 'vif_p', 'gmsd', 
                                         'ms_gmsd', 'vsi', 'haarpsi', 'mdsi']}
    m,std = get_mean_std(params)

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
            adversary = False
            if(adversary):
                tar = t_adv

            
            rmse[i] = weighted_rmse_channels(pred, tar) * std
            acc[i] = weighted_acc_channels(pred-m, tar-m)

            # Compute image similarity metrics
            #for channel in range(n_out_channels):
                #similarity = compute_image_similarity_metrics(pred[:, channel:channel+1, :, :], tar[:, channel:channel+1, :, :])
                #for metric_name, metric_value in similarity.items():
                #    similarity_metrics[metric_name][i, channel] = metric_value

            #if i < prediction_length - 1:
                #print(f'Timestep {i} of {prediction_length}. Channel {idx}:')
            #for metric_name in similarity_metrics:
            #    print(f'  {metric_name}: {similarity_metrics[metric_name][i, idx].item()}')
            
            print('Predicted Unperturbed timestep {} of {}. {} RMS Error: {}, ACC: {}'.format(i, prediction_length, args.field, rmse[i,idx], acc[i,idx]))

            pred = future_pred
            tar = future

    # copy to cpu for plotting/vis
    acc_cpu = acc.cpu().numpy()
    rmse_cpu = rmse.cpu().numpy()
    predictions_cpu = predictions.cpu().numpy()
    targets_cpu = targets.cpu().numpy()

    return acc_cpu, rmse_cpu, predictions_cpu, targets_cpu

#params,data, model, args.prediction_length, idx_vis=params['field_idx'], t_adv=t_adv, epsilon=args.epsilon)
def perturbed_inference(init_delta,params,args,data_slice, model, prediction_length, idx, t_adv, epsilon):
    # create memory for the different stats
    n_out_channels = params['N_out_channels']
    acc = torch.zeros((args.prediction_length, n_out_channels)).to(params.device, dtype=torch.float)
    rmse = torch.zeros((args.prediction_length, n_out_channels)).to(params.device, dtype=torch.float)

    # to conserve GPU mem, only save one channel (can be changed if sufficient GPU mem or move to CPU)
    targets = torch.zeros((args.prediction_length, 1, img_shape_x, img_shape_y)).to(params.device, dtype=torch.float)
    predictions = torch.zeros((args.prediction_length, 1, img_shape_x, img_shape_y)).to(params.device, dtype=torch.float)
    
    similarity_metrics = {metric: torch.zeros((prediction_length, n_out_channels)).to(params.device, dtype=torch.float)
                          for metric in ['psnr','lpips','ssim', 'ms_ssim', 'iw_ssim', 'vif_p', 'gmsd', 
                                         'ms_gmsd', 'vsi', 'haarpsi', 'mdsi']}

    # Apply perturbation to the initial data slice
    perturbed_data_slice = data_slice

    pert_method = 'naive' #['cont','naive','optimal'] 

    if(pert_method == 'cont'):
        pass
        #delta = adversarial_perturbation_continuation(args,data_slice, model, t_adv, prediction_length, epsilon)
    elif(pert_method == 'naive'):
        delta, trajectory= adversarial_perturbation_final_target(args,params,data_slice, model, t_adv, prediction_length, epsilon)
        delta_norm = torch.norm(delta).item()
        print(f"Norm of delta with mse and lpips: {delta_norm}")
    else:
        raise NotImplementedError("Unsupported method for Delta")

    
    # Here you need to send the right init_delta final - prediction length as well otherwise wont work
    delta_opt = init_delta - perturbed_data_slice[0:1]
    delta_norm_opt = torch.norm(delta_opt).item()
    print(f"Norm of optimal delta: {delta_norm_opt}")
    m,std = get_mean_std(params)
    

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
                #import pdb;pdb.set_trace()
                targets[i+1,0] = future[0,idx]

            # compute metrics using the ground truth ERA5 data as "true" predictions
            adversary = False
            if(adversary):
                tar = t_adv
            #import pdb;pdb.set_trace()
            if(i< prediction_length-1):
                rmse[i] = weighted_rmse_channels(pred, tar) * std
                acc[i] = weighted_acc_channels(pred-m, tar-m)
            # Compute image similarity metrics
            #for channel in range(n_out_channels):
                #import pdb;pdb.set_trace()
            #    pred_mean_adjust = pred#-m
            #    tar_mean_adjust = tar #- m
                #similarity = compute_image_similarity_metrics(pred_mean_adjust[:, channel:channel+1, :, :], tar_mean_adjust[:, channel:channel+1, :, :] )
                #for metric_name, metric_value in similarity.items():
                #    similarity_metrics[metric_name][i, channel] = metric_value
            #if i < prediction_length - 1:
                #print(f'Timestep {i} of {prediction_length}. Channel {idx}:')
            #for metric_name in similarity_metrics:
            #    print(f'  {metric_name}: {similarity_metrics[metric_name][i, idx].item()}')
                print('Predicted Perturbed timestep {} of {}. {} RMS Error: {}, ACC: {}'.format(i, prediction_length, args.field, rmse[i,idx], acc[i,idx]))

            pred = future_pred
            tar = future

    # copy to cpu for plotting/vis
    acc_cpu = acc.cpu().numpy()
    rmse_cpu = rmse.cpu().numpy()
    predictions_cpu = predictions.cpu().numpy()
    targets_cpu = targets.cpu().numpy()
    delta = delta.cpu().numpy()

    return acc_cpu, rmse_cpu, predictions_cpu, targets_cpu,trajectory,delta











# def adversarial_perturbation_continuation(args,data_slice, model, t_adv, prediction_length, epsilon, max_iters=100, lr=0.01, loss_func='lpips + mse'):
#     # Initialize perturbation
#     delta = torch.zeros_like(data_slice[0:1], requires_grad=True).to(data_slice.device)
#     optimizer = optim.Adam([delta], lr=lr)
#     n_steps = args.n_steps
#     losses = []

#     # Initialize the LPIPS model if selected (not used in this case since it's MSE)
#     if loss_func == 'lpips + mse':
#         original_lpips_model = LPIPS(net='vgg')
#         custom_lpips_model = CustomLPIPS(original_lpips_model)
#         loss_fn = custom_lpips_model.to(data_slice.device)
#     elif loss_func == 'mse':
#         loss_fn = mse_loss
#     else:
#         raise NotImplementedError("Loss function not implemented")

#     # Current model prediction
#     Z_T = model(data_slice[0:1])
#     for i in range(1, prediction_length):
#         Z_T = model(Z_T)
    
#     # Continuation method loop
#     for step in tqdm(range(1, n_steps + 1), desc="Continuation Steps"):
#         # Compute interpolation parameter s_i
#         s_i = step / n_steps
        
#         # Compute the current target t_i
#         t_i = (1 - s_i) * Z_T + s_i * t_adv
        
#         # Optimize delta to minimize the loss for the current target t_i
#         for _ in tqdm(range(max_iters), desc="Optimization Iterations", leave=False):
#             perturbed_data = data_slice[0:1] + delta
            
#             future_pred = model(perturbed_data)
#             for j in range(1, prediction_length):
#                 future_pred = model(future_pred)
            
#             # Compute loss
#             if loss_func == 'mse':
#                 loss = loss_fn(future_pred, t_i)
#             else:
#                 raise NotImplementedError("Only MSE is currently implemented in this continuation method.")
            
#             losses.append(loss.item())
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             # Apply constraints to delta
#             #with torch.no_grad():
#             #    delta.data = torch.clamp(delta.data, -epsilon, epsilon)
        
#         # Use the computed delta as the initialization for the next step

#     return delta.detach()