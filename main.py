import warnings
import argparse
import os
import torch
import numpy as np
from utils.YParams import YParams
from adversarial_model_utils import load_model, initialize_model
from adversarial_data_utils import load_data,load_data_new, preprocess_data
from model_inference import perturbed_inference,inference
from plot_utils import plot_results, save_rmse_acc_comparison_md, save_tensors_to_pickle,load_tensors_from_pickle
from variables import variables
from plot_utils import visualize_chan
import os

def main(args):

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    torch.cuda.empty_cache()

    
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    #import pdb;pdb.set_trace()

    params = YParams(args.config_file, args.config_name)

    # in and out channels: FourCastNet uses 20 input channels corresponding to 20 prognostic variables
    in_channels = np.array(params.in_channels)
    out_channels = np.array(params.out_channels)
    params['N_in_channels'] = len(in_channels)
    params['N_out_channels'] = len(out_channels)
    #import pdb;pdb.set_trace()
    params.means = np.load(args.global_means_path)[0, out_channels]  # for normalizing data with precomputed train stats
    params.stds = np.load(args.global_stds_path)[0, out_channels]
    params.time_means = np.load(args.time_means_path)[0, out_channels]
    
    ## Use cuda 0,1 only
    params.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu') 
    #import pdb;pdb.set_trace()

    # Load and prepare the model
    model = initialize_model(params, params.device, args.model_path)

    #import pdb;pdb.set_trace()
   

    ### Smaller Data Segments Prediction Length 4
    data, t_adv = load_data(args.data_file, params, args.field, args.prediction_length)
    data = load_data_new(args.data_file, params, args.field,1,5) #6:10,1:5
    data = preprocess_data(data, params)
    ## Let us try and visualize this data for our purposes. It accepts normalized data
    # 2 corresponds to temperature
    #visualize_chan(data,params,channel_vis=2,norm = True) 
    #import pdb;pdb.set_trace()
    t_adv = load_data_new(args.data_file, params, args.field,19,20)
    t_adv = preprocess_data(t_adv, params)
    init_delta = load_data_new(args.data_file, params,args.field,16,17) # 9,10,16,17
    init_delta = preprocess_data(init_delta, params)


    #### Larger Data Segments Prediction Length 9
    # data = load_data_new(args.data_file, params, args.field,0,20) #6:10,1:5
    # data = preprocess_data(data, params)
    # t_adv = load_data_new(args.data_file, params, args.field,19,20)
    # t_adv = preprocess_data(t_adv, params)
    # init_delta = load_data_new(args.data_file, params,args.field,10,11) # 9,10,16,17
    # init_delta = preprocess_data(init_delta, params)

   
    
    # Visualization
    idx_vis = variables.index(args.field) # also prints out metrics for this field
    #args.channel_perturb = [idx_vis]
    #args.channel_perturb = [2, 5, 15, 17, 18, 19] # pertrubing correlated variables
    args.channel_perturb  = [i for i in range(20)]
    #args.channel_perturb = idx_vis
    #import pdb;pdb.set_trace()

    visualization = False
    delta_opt = True
    
    if(delta_opt):
    # Run inference
        acc_cpu, rmse_cpu, predictions_cpu, targets_cpu = inference(params,args,data, model, args.prediction_length, idx_vis,t_adv=t_adv)
        
        acc_p_cpu, rmse_p_cpu, perturbed_predictions_cpu, targets_cpu, trajectory,delta = perturbed_inference(init_delta,params,args,
         data, model, args.prediction_length, idx_vis, t_adv,args.epsilon)

    

        save_tensors_to_pickle(delta,acc_p_cpu, rmse_p_cpu, perturbed_predictions_cpu, targets_cpu,trajectory,args)


    # Visualize previous results
    if(visualization):
        
        pickle_file_path = os.path.join(args.output_path, "saved_tensors_mse", "tensors.pkl")
    
        # Load the tensors
        tensors = load_tensors_from_pickle(pickle_file_path)

        # Access the tensors
        acc_p_cpu = tensors["acc_p_cpu"]
        rmse_p_cpu = tensors["rmse_p_cpu"]
        perturbed_predictions_cpu = tensors["perturbed_predictions_cpu"]
        targets_cpu = tensors["targets_cpu"]
        trajectory = tensors["trajectory"]
        delta = tensors["delta"]

        
        print(f'Norm of delta: {np.linalg.norm(delta)}')

        ## Optimal Delta

        delta_opt = init_delta - data[0:1]
        delta_norm_opt = torch.norm(delta_opt).item()
        print(f"Norm of optimal delta: {delta_norm_opt}")
        import pdb;pdb.set_trace()

        acc_cpu, rmse_cpu, predictions_cpu, targets_cpu = inference(params,args,data, model, args.prediction_length, idx_vis,t_adv=t_adv)
    
    
    plot_results(params,predictions_cpu, targets_cpu, perturbed_predictions_cpu, t_adv, acc_cpu, rmse_cpu, acc_p_cpu, rmse_p_cpu, args)
    # Save table
    #save_rmse_acc_comparison_md(acc_p_cpu, rmse_p_cpu, acc_cpu, rmse_cpu)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adversarial Perturbation on FourCastNet')

    # Default paths
    default_config_file = "./config/AFNO.yaml"
    default_config_name = "afno_backbone"
    default_data_file = "./FourCast_Dataset/data/FCN_ERA5_data_v0/out_of_sample/2018.h5"
    default_model_path = "./FourCast_Dataset/model_weights/FCN_weights_v0/backbone.ckpt"
    default_global_means_path = "./FourCast_Dataset/additional/stats_v0/global_means.npy"
    default_global_stds_path = "./FourCast_Dataset/additional/stats_v0/global_stds.npy"
    default_time_means_path = "./FourCast_Dataset/additional/stats_v0/time_means.npy"
    default_land_sea_mask_path = "./FourCast_Dataset/additional/stats_v0/land_sea_mask.npy"
    default_field = "sp"
    default_gamma = 1 # Contribution of MSE Loss
    default_lamda = 0 # Contribution of LPIPS loss
    #default_output_path = "./results" 
    default_output_path = "./surface_pressure" #./perturbation_channels_only"#"./patch_smooth_perturbation"#"./patch_smooth_perturbation" #"./perturb_results" #"./new_results"  # ./new_results" 
    default_tv_weight = torch.ones(20) * 0.01  # Default: 1x20 vector with 0.1 for each channel
    default_linfty_weight = torch.ones(20) * 0.01  # Default: 1x20 vector with 0.1 for each channel

      # Add two new arguments for inftyWeight and tv
    default_infty_loss = [9.1473e-08, 2.5559e-06, 1.3177e-06, 2.8082e-07, 1.8507e-06, 4.2667e-07,
                          1.2589e-07, 5.4461e-07, 1.9669e-06, 5.8189e-07, 1.7013e-06, 3.5134e-06,
                          3.3896e-07, 1.8124e-07, 2.4146e-06, 1.8668e+00, 3.6435e-06, 4.8708e-06,
                          1.6442e-06, 6.4423e-08]
    default_tv_loss = [99165.6016, 109694.2422, 27920.4980, 58921.9922, 34445.3281,
                       22821.9902, 98763.7188, 108050.5078, 32448.5762, 89158.7656,
                       102990.0625, 20403.5078, 61842.3750, 63398.3281, 13680.4805,
                       19226.8984, 9923.5332, 114838.1641, 142137.9844, 49321.4219]
    

    ### Change Africa Temperature M:[300,600], L:[1100 1300]
    #1, N, M (Rows), L (Columns) --> 1,20,720 ,1440

    parser.add_argument('--method')
    parser.add_argument('--config_file', type=str, default=default_config_file, help='Path to the config file.')
    parser.add_argument('--config_name', type=str, default=default_config_name, help='Name of the config in the yaml file.')
    parser.add_argument('--data_file', type=str, default=default_data_file, help='Path to the data file.')
    parser.add_argument('--model_path', type=str, default=default_model_path, help='Path to the model checkpoint.')
    parser.add_argument('--global_means_path', type=str, default=default_global_means_path, help='Path to the global means file.')
    parser.add_argument('--global_stds_path', type=str, default=default_global_stds_path, help='Path to the global stds file.')
    parser.add_argument('--time_means_path', type=str, default=default_time_means_path, help='Path to the time means file.')
    parser.add_argument('--land_sea_mask_path', type=str, default=default_land_sea_mask_path, help='Path to the land-sea mask file.')
    parser.add_argument('--field', type=str, default=default_field, help='Field to visualize (e.g., "t2m").')
    parser.add_argument('--prediction_length', type=int, default=4, help='Length of prediction.')
    parser.add_argument('--gamma', type=float, default=default_gamma, help='MSE Loss Contribution.')
    parser.add_argument('--n_steps', type=int, default=20, help='Continuation Method:Number of Steps')
    parser.add_argument('--lamda', type=float, default=default_lamda, help='LPIPS loss Contribution.')
    parser.add_argument('--output_path', type=str, default=default_output_path, help='Path to save the output plots.')
    # New arguments for channel mask and patch mask with default values
    parser.add_argument('--channel_perturb', type=int, default=2, help='Channel to apply perturbation (default: 0).')
    parser.add_argument('--patch_start_L', type=int, default=1100, help='Starting point for the patch in the L direction (default: 0).')
    parser.add_argument('--patch_start_M', type=int, default=300, help='Starting point for the patch in the M direction (default: 0).')
    parser.add_argument('--patch_size_L', type=int, default=200, help='Size of the patch in the L direction (default: 10).')
    parser.add_argument('--patch_size_M', type=int, default=300, help='Size of the patch in the M direction (default: 10).')
    parser.add_argument('--tv_weight', type=float, nargs=20, default=default_tv_weight.tolist(), 
                        help='Weights for Total Variation penalty, one per channel (default: 0.1 for each channel).')
    parser.add_argument('--linfty_weight', type=float, nargs=20, default=default_linfty_weight.tolist(),
                        help='Weights for L_infty penalty, one per channel (default: 0.1 for each channel).')
  
    
    parser.add_argument('--epsilon', type=float, nargs=20, default=default_infty_loss, 
                        help='Default values for L_infty loss across 20 channels.')
    parser.add_argument('--tau', type=float, nargs=20, default=default_tv_loss, 
                        help='Default values for Total Variation loss across 20 channels.')

    args = parser.parse_args()

    # Ensure tv_weight, linfty_weight, inftyWeight, and tv are tensors
    args.tv_weight = torch.tensor(args.tv_weight)
    args.linfty_weight = torch.tensor(args.linfty_weight)
    args.epsilon = torch.tensor(args.epsilon)
    args.tau = torch.tensor(args.tau)

    args.log_dir = default_output_path




    main(args)
