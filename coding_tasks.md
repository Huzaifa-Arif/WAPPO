# TO DO Experimentation

- Does the basic work? Position of optimizer.step()?
- 







# TO DO Recommendation

- MSE is the only objective
- Make certain fields sparse and see the effect
- Ablation and have only one parameter
- Violations of constraints ie: are the outputs plausible?
- Norm constrainesd
- Change the objective to weighted RMSE and observe its impact.
- Change the loss function from L2 to L-\(\infty\)
- Investigate the convergence of the adversarial perturbation with different constraints.
- See with more iterations

# TO DOs (Huzaifa)
- Implement the continuation method and observe if the perturbation improves the output.
- Show the differences between:
  - a) Perturbed Output
  - b) Ground truth
  - c) Adversarial targets
  - with different initialization methods (zero and continuation method).
- Fourier Perturbations and their impact.
- Change the objective to weighted RMSE and observe its impact.
- LPIPS ablation:
  - Normalize the input and observe the improvement in the visualization with [-1,1] normalization.
  - Consider 3 channels at a time and average the loss for 20 channels.
- A lot of Parameters need to be in the arg space

### Non-Urgent Coding Challenges
- Change the loss function from L2 to L-\(\infty\) and L-1 and observe the effect of this ablation.
- Implement gradient checkpointing, sharding in LLM inference, or consider using ZOO (worst-case scenario) if CUDA challenges persist (see below for details).
- Increase prediction length (an additional empirical point).
- Investigate the convergence of the adversarial perturbation with different constraints.
- Increase the number of iterations.

### Minor Coding Improvements and Understanding
- Clarify the meaning of `m` and `std`.
- Review the visualization of channels.
- Ensure that temperature considerations are applied only for visualization purposes.

### Improve the writing of some sections



### Copying results : Run them from the terminal you want. Note: It works only for Desktop PC
scp -r arifh@sunrise.cs.rpi.edu:/home/arifh/ForecastNet/FourCastNet/results /home/arifh/Desktop/WeatherForecast




