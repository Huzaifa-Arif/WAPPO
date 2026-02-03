import os
import pickle
import matplotlib.pyplot as plt
from plot_utils import plot_trajectory_delta,plot_trajectory_all

def load_tensors_from_pickle(pickle_file_path):
    with open(pickle_file_path, 'rb') as f:
        tensors = pickle.load(f)
    return tensors

class Args:
    def __init__(self, output_path, field, prediction_length):
        self.output_path = output_path
        self.field = field
        self.prediction_length = prediction_length

path = "./new_results"
# Define the path to the pickle file
pickle_file_path = os.path.join(path, "saved_tensors", "tensors.pkl")
    
# Load the tensors
tensors = load_tensors_from_pickle(pickle_file_path)

# Access the tensors
perturbed_predictions_cpu = tensors["perturbed_predictions_cpu"]
targets_cpu = tensors["targets_cpu"]
trajectory = tensors["trajectory"]

args = Args(
    output_path= "./new_results",
    field="t2m",  # or another field relevant to your dataset
    prediction_length=4  # or whatever your prediction length is
)

# Plot the trajectory of the delta perturbations
plot_trajectory_delta(args, trajectory)

# Plot the comparison of ground truth vs. perturbed predictions
plot_trajectory_all(targets_cpu, perturbed_predictions_cpu, t_adv=None, args=args)
