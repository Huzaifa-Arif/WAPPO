import h5py
import numpy as np
import os, sys, time
import torch
import os
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.optim as optim
import pytorch_lightning as pl
from utils.YParams import YParams
from networks.afnonet import AFNONet
import sys


torch.set_float32_matmul_precision('medium')



# d
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


# define metrics from the definitions above
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

class WeatherForecastDataset(Dataset):
    def __init__(self, data_file, in_channels, means, stds):
        self.data_file = data_file
        self.in_channels = in_channels
        self.means = means
        self.stds = stds
        
        with h5py.File(data_file, 'r') as f:
            self.data_length = f['fields'].shape[0]
    
    def __len__(self):
        return self.data_length
    
    def __getitem__(self, idx):
        with h5py.File(self.data_file, 'r') as f:
            data = f['fields'][idx, self.in_channels, 0:720]
        
        data = (data - self.means) / self.stds
        
        return torch.tensor(data, dtype=torch.float32)




class AdversarialAttack(pl.LightningModule):
    def __init__(self, pretrained_model, t_adv,initial_condition, epsilon, lr=0.001, prediction_length=10, img_shape_x=720, img_shape_y=1440, n_out_channels=20):
        super(AdversarialAttack, self).__init__()
        self.pretrained_model = pretrained_model
        self.t_adv = t_adv  # Target adversarial sample
        self.epsilon = epsilon  # Maximum perturbation
        self.lr = lr
        self.initial_condition = initial_condition
        self.prediction_length = prediction_length  # Number of future steps to predict
        self.img_shape_x = img_shape_x
        self.img_shape_y = img_shape_y
        self.n_out_channels = n_out_channels
        self.perturbation = nn.Parameter(torch.zeros(1, len(in_channels), img_shape_x, img_shape_y), requires_grad=True)  # Initialize perturbation
        #self.perturbation = nn.Parameter(torch.zeros(sequence_length, len(in_channels), img_shape_x, img_shape_y), requires_grad=True)  # Initialize perturbation
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
    def forward(self, x):
        # Apply perturbation to the input
        perturbed_data = self.initial_condition + self.perturbation
        perturbed_data = torch.clamp(perturbed_data, -self.epsilon, self.epsilon)  # Ensure the perturbed input is valid
        
        # Initialize the sequence with perturbed input
        current_input = perturbed_data
        model_output = []
        
        with torch.no_grad():
            for i in range(self.prediction_length):
                if i == 0:
                    future_pred = self.pretrained_model(perturbed_data[0:1])
                else:
                    future_pred = self.pretrained_model(future_pred)
                #print("shape:", future_pred.shape)
                model_output.append(future_pred)
        #sys.exit()
        # model_output = torch.stack(model_output).squeeze()

        return future_pred

    def training_step(self, batch, batch_idx):
        x = batch
        y_hat = self(x)
        
        # The goal is to make the final prediction close to the target adversarial sample
        #print("Sample:",y_hat[-1].shape)
        #print("Predict:",y_hat.shape)
        #print("Adversary:",self.t_adv.shape)
        #sys.exit()
        #import pdb; pdb.set_trace()
        loss = nn.MSELoss()(y_hat, self.t_adv)
        self.log('train_loss', loss)
        
        # Apply the gradient update to the perturbation only
        return {'loss': loss}
    
    def configure_optimizers(self):
        return torch.optim.Adam([self.perturbation], lr=self.lr)
    
    def on_fit_start(self):
        self.t_adv = self.t_adv.to(self.device)
        self.initial_condition = self.initial_condition.to(self.device)

    
    def test_step(self, batch, batch_idx):
        x = batch
        
        # Perform perturbed inference
        acc_perturbed, rmse_perturbed, predictions_perturbed, targets_perturbed = self.perturbed_inference(x, self.pretrained_model, self.prediction_length, self.t_adv, True)
        
        # Perform unperturbed inference
        acc_unperturbed, rmse_unperturbed, predictions_unperturbed, targets_unperturbed = self.perturbed_inference(x, self.pretrained_model, self.prediction_length, self.t_adv, False)
        
        # Log results
        self.log('test_rmse_perturbed', rmse_perturbed.mean())
        self.log('test_rmse_unperturbed', rmse_unperturbed.mean())
        self.log('test_acc_perturbed', acc_perturbed.mean())
        self.log('test_acc_unperturbed', acc_unperturbed.mean())
        
        return {
            'acc_perturbed': acc_perturbed,
            'rmse_perturbed': rmse_perturbed,
            'predictions_perturbed': predictions_perturbed,
            'targets_perturbed': targets_perturbed,
            'acc_unperturbed': acc_unperturbed,
            'rmse_unperturbed': rmse_unperturbed,
            'predictions_unperturbed': predictions_unperturbed,
            'targets_unperturbed': targets_unperturbed
        }
    
    def perturbed_inference(self, data, model, prediction_length, t_adv, perturb = False):
        acc = torch.zeros((prediction_length, self.n_out_channels))
        rmse = torch.zeros((prediction_length, self.n_out_channels))
        targets = torch.zeros((prediction_length, self.n_out_channels, self.img_shape_x, self.img_shape_y))
        predictions = torch.zeros((prediction_length, self.n_out_channels, self.img_shape_x, self.img_shape_y))

        if(perturb):
            delta = self.perturbation
        else:
            delta = 0

        perturbed_data_slice = self.initial_condition + delta

        with torch.no_grad():
            for i in range(prediction_length):
                if i == 0:
                    first = data[0:1]
                    future = data[1:2]
                    pred = first
                    tar = first
                    #targets[0] = first[0]
                    #predictions[0] = first[0]
                    future_pred = model(first)
                else:
                    if i < prediction_length - 1:
                        future = data[i+1:i+2]
                    future_pred = model(future_pred)

                if i < prediction_length - 1:
                    predictions[i+1] = future_pred[0]
                    targets[i+1] = future_pred[0]

                # compute metrics using the ground truth ERA5 data as "true" predictions
                rmse[i] = weighted_rmse_channels(pred, tar) * std
                acc[i] = weighted_acc_channels(pred-m, tar-m)
                field = 't2m'
                print('Predicted Unperturbed timestep {} of {}. {} RMS Error: {}, ACC: {}'.format(i, prediction_length, field, rmse[i], acc[i]))

            pred = future_pred
            tar = future

        acc_cpu = acc.cpu().numpy()
        rmse_cpu = rmse.cpu().numpy()
        predictions_cpu = predictions.cpu().numpy()
        targets_cpu = targets.cpu().numpy()

        return acc_cpu, rmse_cpu, predictions_cpu, targets_cpu

    


if __name__ == '__main__':


   # Data and model paths
    data_path = "./FourCast_Dataset/data/FCN_ERA5_data_v0/out_of_sample"
    data_file = os.path.join(data_path, "2018.h5")
    model_path = "./FourCast_Dataset/model_weights/FCN_weights_v0/backbone.ckpt"
    global_means_path = "./FourCast_Dataset/additional/stats_v0/global_means.npy"
    global_stds_path = "./FourCast_Dataset/additional/stats_v0/global_stds.npy"
    time_means_path = "./FourCast_Dataset/additional/stats_v0/time_means.npy"
    land_sea_mask_path = "./FourCast_Dataset/additional/stats_v0/land_sea_mask.npy"

     # Default config
    config_file = "./config/AFNO.yaml"
    config_name = "afno_backbone"
    params = YParams(config_file, config_name)


    # in and out channels: FourCastNet uses 20 input channels corresponding to 20 prognostic variables
    in_channels = np.array(params.in_channels)
    out_channels = np.array(params.out_channels)
    params['N_in_channels'] = len(in_channels)
    params['N_out_channels'] = len(out_channels)
    params.means = np.load(global_means_path)[0, out_channels] # for normalizing data with precomputed train stats
    params.stds = np.load(global_stds_path)[0, out_channels]
    params.time_means = np.load(time_means_path)[0, out_channels]


   
    if params.nettype == 'afno':
     model = AFNONet(params)
    else:
      raise Exception("not implemented")
    

    model = load_model(model, params, model_path)

  


    img_shape_x = 720
    img_shape_y = 1440
    means = params.means
    stds = params.stds
    time_means = params.time_means
    
    m = torch.as_tensor((time_means - means)/stds)[:, 0:img_shape_x]
    m = torch.unsqueeze(m, 0)
    std = torch.as_tensor(stds[:,0,0])

     # Example usage
    
    epsilon = 0.1  # Maximum allowed perturbation
    lr = 0.01  # Learning rate for optimizing perturbation
    prediction_length = 2  # Number of steps to predict
    n_out_channels = 20

    #Load the initial condition
    with h5py.File(data_file, 'r') as f:
        initial_condition = f['fields'][0:1, in_channels, 0:img_shape_x]
    initial_condition = (initial_condition - means) / stds
    initial_condition = torch.tensor(initial_condition, dtype=torch.float32)



   

    # Load the 20th time step as t_adv
    with h5py.File(data_file, 'r') as f:
        t_adv = f['fields'][19:20, in_channels, 0:img_shape_x]
    t_adv = (t_adv - means) / stds
    t_adv = torch.tensor(t_adv, dtype=torch.float32)

    #print("t_adv_shape:",t_adv.shape)
    #sys.exit()

    model = AdversarialAttack(model, t_adv, initial_condition, epsilon, lr, prediction_length, img_shape_x, img_shape_y, n_out_channels)
    #trainer = pl.Trainer(max_epochs=50)
    trainer = pl.Trainer(max_epochs=50, devices=1, accelerator="gpu", strategy="ddp",log_every_n_steps=10)

    #trainer = pl.Trainer(max_epochs=50, gpus=2, strategy="ddp")



    # Create the dataset and dataloader
    test_dataset = WeatherForecastDataset(data_file, in_channels, means, stds)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=63)


    # for i, data in enumerate(test_loader):
    #     print(f"Batch {i+1}:")
    #     print(f"Data shape: {data.shape}")
    
   
    #     #if i >= 5:  # For example, inspect only the first 5 batches
    #     #    break
    

    #test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    trainer.fit(model, test_loader)
    
    print("Optimal perturbation found")
    sys.exit()

    # Test the model
    test_results = trainer.test(model, test_loader)



  
#### Modify so that the pretrained model does not have its weights modified