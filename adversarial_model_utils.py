import torch
from networks.afnonet import AFNONet
from collections import OrderedDict
import lpips
import torch.nn as nn
from torchmetrics.functional.image import total_variation

def apply_channel_mask(delta, channel, num_channels):
    """
    Apply delta only to the specified channel.
    
    :param delta: Perturbation tensor (shape: [1, N, M, L])
    :param channel: The channel index to which the perturbation should be applied
    :param num_channels: Total number of channels in the data
    :return: Perturbation applied only to the specified channel
    """
    # Create a channel mask with shape [1, N, 1, 1], where N is the number of channels
    channel_mask = torch.zeros(delta.shape).to(delta.device) # Mask for channels

    #import pdb;pdb.set_trace()
    
    for ch in channel:
     channel_mask[:, ch, :, :] = 1

    
    # Apply the mask to delta (only applies delta to the specified channel)
    delta_masked = delta * channel_mask

    
    
    return delta_masked

def apply_patch_mask(delta, patch_start_L, patch_start_M, patch_size_L, patch_size_M):
    """
    Apply patch mask to the spatial dimensions of delta.

    :param delta: Perturbation tensor (shape: [1, N, M, L])
    :param patch_start_L: Starting point for the patch in the L direction
    :param patch_start_M: Starting point for the patch in the M direction
    :param patch_size_L: Size of the patch in the L direction
    :param patch_size_M: Size of the patch in the M direction
    :return: Perturbation applied only to the specified patch region
    """
    # Create a patch mask of shape [1, 1, M, L]
    patch_mask = torch.zeros(1, 1, delta.shape[2], delta.shape[3]).to(delta.device)
    
    # Set the patch region to 1 in the mask
    patch_mask[:, :, patch_start_M:patch_start_M+patch_size_M, patch_start_L:patch_start_L+patch_size_L] = 1
    
    # Apply the mask to delta (only applies delta to the specified patch region)
    patch_mask = patch_mask
    
    return patch_mask


def initialize_model(params, device, model_path):
    # Initialize the model and move it to the appropriate device
    if params.nettype == 'afno':
        model = AFNONet(params).to(device)
    else:
        raise Exception("Model type not implemented")
    
    # Load the pretrained model weights
    model = load_model(model, params, model_path)
    
    return model

def load_model(model, params, checkpoint_file):
    # Load the checkpoint and map it to the correct device (CPU or GPU)
    if params.device.type == 'cpu':
        checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_file)
    
    try:
        # If the model was trained with DDP, strip the 'module.' prefix from the keys
        new_state_dict = OrderedDict()
        for key, val in checkpoint['model_state'].items():
            name = key[7:]  # Strip 'module.' prefix
            if name != 'ged':  # Skip the 'ged' key if it exists
                new_state_dict[name] = val
        model.load_state_dict(new_state_dict)
    except:
        # If there is no 'module.' prefix, load the state dict directly
        model.load_state_dict(checkpoint['model_state'])

    # Set the model to evaluation mode for inference
    model.eval()
    
    return model

# Custom LPIPS model to handle 20-channel inputs
# Custom LPIPS model to handle 20-channel inputs
class CustomLPIPS(nn.Module):
    def __init__(self, original_lpips_model):
        super(CustomLPIPS, self).__init__()
        self.lpips_model = original_lpips_model
        # Modify the first convolution layer to accept 20 channels
        self.lpips_model.net = self.modify_first_layer(self.lpips_model.net)
        # Modify the scaling layer to handle 20 channels
        self.lpips_model.scaling_layer = self.modify_scaling_layer(self.lpips_model.scaling_layer)

    def modify_first_layer(self, net):
        # Recursively find and modify the first Conv2d layer
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                # Create a new Conv2d layer with 20 input channels
                new_conv = nn.Conv2d(20, module.out_channels, kernel_size=module.kernel_size, 
                                     stride=module.stride, padding=module.padding)
                
                # Copy the weights of the first 3 channels from the original conv layer
                with torch.no_grad():
                    new_conv.weight[:, :3, :, :] = module.weight[:, :3, :, :]
                    if new_conv.in_channels > 3:
                        # Initialize the remaining channels
                        nn.init.kaiming_normal_(new_conv.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')
                
                # Copy the bias if it exists
                if module.bias is not None:
                    new_conv.bias = nn.Parameter(module.bias.clone())

                # Replace the original conv layer with the new one
                parent_module = net
                for part in name.split('.')[:-1]:
                    parent_module = getattr(parent_module, part)
                setattr(parent_module, name.split('.')[-1], new_conv)
                break
        return net

    def modify_scaling_layer(self, scaling_layer):
        # Custom scaling for 20 channels
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

        new_shift = torch.tensor([0.5] * 20).view(1, 20, 1, 1).to(device)
        new_scale = torch.tensor([0.5] * 20).view(1, 20, 1, 1).to(device)

        
        # Update scaling layer to work with 20 channels
        class CustomScalingLayer(nn.Module):
            def forward(self, inp):
                return (inp - new_shift) / new_scale
        
        return CustomScalingLayer()

    def forward(self, x, y):
        return self.lpips_model(x, y)



class HsLossWithPerturbations(object):
    def __init__(self, d=2, p=2, k=1, a=None, group=False, size_average=True, reduction=True):
        super(HsLossWithPerturbations, self).__init__()

        # Dimension and Lp-norm type are positive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average

        if a is None:
            a = [1,] * k
        self.a = a

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)
        return diff_norms / y_norms

    def __call__(self, x, y, delta_low=None, delta_high=None, mask_low=None, mask_high=None):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        # Fourier transform of x and y
        x_fourier = torch.fft.fftn(x, dim=[1, 2])
        y_fourier = torch.fft.fftn(y, dim=[1, 2])

        # Apply perturbations in the Fourier domain
        if delta_low is not None and mask_low is not None:
            x_fourier = x_fourier + delta_low * mask_low
        if delta_high is not None and mask_high is not None:
            x_fourier = x_fourier + delta_high * mask_high

        # Generate frequency grids
        k_x = torch.cat((torch.arange(start=0, end=nx//2, step=1),
                         torch.arange(start=-nx//2, end=0, step=1)), 0).reshape(nx, 1).repeat(1, ny)
        k_y = torch.cat((torch.arange(start=0, end=ny//2, step=1),
                         torch.arange(start=-ny//2, end=0, step=1)), 0).reshape(1, ny).repeat(nx, 1)
        k_x = torch.abs(k_x).reshape(1, nx, ny, 1).to(x.device)
        k_y = torch.abs(k_y).reshape(1, nx, ny, 1).to(x.device)

        if not balanced:
            weight = 1
            if k >= 1:
                weight += a[0]**2 * (k_x**2 + k_y**2)
            if k >= 2:
                weight += a[1]**2 * (k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
            weight = torch.sqrt(weight)
            loss = self.rel(x_fourier * weight, y_fourier * weight)
        else:
            loss = self.rel(x_fourier, y_fourier)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x**2 + k_y**2)
                loss += self.rel(x_fourier * weight, y_fourier * weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
                loss += self.rel(x_fourier * weight, y_fourier * weight)
            loss = loss / (k + 1)

        return loss



class MSELossWithFourierPerturbations(nn.Module):
    def __init__(self):
        super(MSELossWithFourierPerturbations, self).__init__()

    def forward(self, x, y, delta_low=None, delta_high=None, mask_low=None, mask_high=None):
        # Perform Fourier transform on the input
        x_fourier = torch.fft.fftn(x, dim=[1, 2])

        # Apply perturbations in the Fourier space
        if delta_low is not None and mask_low is not None:
            x_fourier = x_fourier + delta_low * mask_low
        if delta_high is not None and mask_high is not None:
            x_fourier = x_fourier + delta_high * mask_high

        # Inverse Fourier transform to get back to the spatial domain
        x_perturbed = torch.fft.ifftn(x_fourier, dim=[1, 2])

        # Ensure the result is in the real domain
        x_perturbed = x_perturbed.real

        # Calculate MSE between the perturbed prediction and the target
        mse_loss = torch.mean((x_perturbed - y) ** 2)

        return mse_loss


def total_variation_per_channel(x):
    """
    Compute Total Variation loss for each channel in the tensor.
    
    :param x: Tensor of shape [B, C, H, W] (Batch, Channels, Height, Width)
    :return: Tensor of shape [C] representing TV loss for each channel
    """
    batch_size, num_channels, _, _ = x.shape
    tv_loss_per_channel = torch.zeros(num_channels).to(x.device)
    
    # Iterate over each channel and compute TV loss
    for c in range(num_channels):
        tv_loss_per_channel[c] = total_variation(x[:, c:c+1, :, :])  # Compute TV loss for channel c
    
    return tv_loss_per_channel