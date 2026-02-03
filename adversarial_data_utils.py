import h5py
import torch

def load_data(data_file, params, field, prediction_length):
    img_shape_x = 720
    with h5py.File(data_file, 'r') as f:
        data = f['fields'][:prediction_length, params.in_channels,0:img_shape_x]
        t_adv = f['fields'][19:20, params.in_channels,0:img_shape_x]
    return data, t_adv

def load_data_new(data_file, params, field,start,end):
    img_shape_x = 720
    with h5py.File(data_file, 'r') as f:
        data = f['fields'][start:end, params.in_channels,0:img_shape_x]
    return data

def preprocess_data(data, params):
    img_shape_x = 720
    img_shape_y = 1440

    # means and stds over training data
    means = params.means
    stds = params.stds

    # load climatological means
    time_means = params.time_means # temporal mean (for every pixel) this is tensor
    m = torch.as_tensor((time_means - means)/stds)[:, 0:img_shape_x]
    m = torch.unsqueeze(m, 0)
    # these are needed to compute ACC and RMSE metrics
    m = m.to(params.device, dtype=torch.float)
    std = torch.as_tensor(stds[:,0,0]).to(params.device, dtype=torch.float)
    data = (data - means)/stds
    data = torch.as_tensor(data).to(params.device, dtype=torch.float)
    return data
