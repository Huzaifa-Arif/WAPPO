import torch

import torch
import piq
from piq import (
    ssim, SSIMLoss, multi_scale_ssim, information_weighted_ssim, vif_p,
    fsim, gmsd, multi_scale_gmsd, vsi, haarpsi, mdsi, psnr
)

def apply_latitude_weights(tensor, num_lat):
    """
    Apply latitude weighting to a tensor of shape (N, C, H, W).
    Args:
        tensor (torch.Tensor): Input tensor of shape (N, C, H, W).
        num_lat (int): Number of latitude points (H dimension).
    Returns:
        torch.Tensor: Weighted tensor of the same shape as input.
    """
    lat_t = torch.arange(start=0, end=num_lat, device=tensor.device)
    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = latitude_weighting_factor(lat_t, num_lat, s).view(1, 1, -1, 1)
    return tensor * weight

def compute_image_similarity_metrics(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0):
    """
    Compute various image similarity metrics between two batches of images,
    applying latitude weighting to account for the Earth's curvature.
    
    Args:
        x (torch.Tensor): An input tensor of shape (N, C, H, W).
        y (torch.Tensor): A target tensor of shape (N, C, H, W).
        data_range (float): Maximum value range of images (usually 1.0 or 255).
        
    Returns:
        dict: A dictionary containing the computed metrics.
    """
    num_lat = x.shape[2]
    
    # Apply latitude weights
    x_weighted = apply_latitude_weights(x, num_lat)
    y_weighted = apply_latitude_weights(y, num_lat)


    x_weighted = torch.clamp(x_weighted, 0, data_range)
    y_weighted = torch.clamp(y_weighted, 0, data_range)
    
    metrics = {}
    
    # PSNR
    metrics['psnr'] = psnr(x_weighted, y_weighted, data_range=data_range)

    #LPIPS
    lpips_loss = piq.LPIPS(reduction='none')(x_weighted, y_weighted)
    metrics['lpips'] = lpips_loss
    
    # SSIM
    metrics['ssim'] = ssim(x_weighted, y_weighted, data_range=data_range)
    
    # MS-SSIM
    metrics['ms_ssim'] = multi_scale_ssim(x_weighted, y_weighted, data_range=data_range)
    
    # IW-SSIM
    metrics['iw_ssim'] = information_weighted_ssim(x_weighted, y_weighted, data_range=data_range)
    
    # VIFp
    metrics['vif_p'] = vif_p(x_weighted, y_weighted, data_range=data_range)
    
    # GMSD
    metrics['gmsd'] = gmsd(x_weighted, y_weighted, data_range=data_range)
    
    # MS-GMSD
    metrics['ms_gmsd'] = multi_scale_gmsd(x_weighted, y_weighted, data_range=data_range)
    
    # VSI
    metrics['vsi'] = vsi(x_weighted, y_weighted, data_range=data_range)
    
    # HaarPSI
    metrics['haarpsi'] = haarpsi(x_weighted, y_weighted, data_range=data_range)
    
    # MDSI
    metrics['mdsi'] = mdsi(x_weighted, y_weighted, data_range=data_range)
    
    return metrics





def lat(j, num_lat):
    return 90. - j * 180./float(num_lat-1)

def latitude_weighting_factor(j, num_lat, s):
    return num_lat * torch.cos(3.1416/180. * lat(j, num_lat))/s

def weighted_rmse_channels(pred, target):
    num_lat = pred.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor(lat_t, num_lat, s), (1, 1, -1, 1))
    return torch.sqrt(torch.mean(weight * (pred - target)**2., dim=(-1,-2)))

def weighted_acc_channels(pred, target):
    num_lat = pred.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor(lat_t, num_lat, s), (1, 1, -1, 1))
    return torch.sum(weight * pred * target, dim=(-1,-2)) / torch.sqrt(torch.sum(weight * pred * pred, dim=(-1,-2)) * torch.sum(weight * target * target, dim=(-1,-2)))
