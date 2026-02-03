import torch
import torch.optim as optim

def adversarial_perturbation_final_target(data_slice, model, t_adv, prediction_length, epsilon, max_iters=100, lr=0.01):
    delta = torch.zeros_like(data_slice, requires_grad=True).to(data_slice.device)
    optimizer = optim.Adam([delta], lr=lr)

    for _ in range(max_iters):
        perturbed_data = data_slice + delta
        perturbed_data = torch.clamp(perturbed_data, 0, 1)

        perturbed_data.requires_grad_()
        sample_length = 1

        sampled_outputs = []
        for _ in range(sample_length):
            future_pred = model(perturbed_data[0:1])
            for i in range(1, prediction_length):
                future_pred = model(future_pred)
            sampled_outputs.append(future_pred)

        sampled_outputs = torch.stack(sampled_outputs)
        loss = torch.nn.functional.mse_loss(sampled_outputs.mean(dim=0), t_adv)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)

    return delta.detach()

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
