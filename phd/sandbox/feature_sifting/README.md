Experiments I want to run:
- How does LMS perform (sweep step-size)?
- How does Autostep perform (sweep step-size and meta step-size)?
- How does continual backprop perform (sweep step-size, replacement rate, and utility ema decay rate)?
- Test Fable method (sweep step-size and maybe something else)?

Notebook structure:
- Imports
- Define portion of sweep config that is unchanging (experiments, optuna_storage, mlflow_storage, output_dir, spec, algorithm), and uses optuna and mlflow local file storage relative to the notebook
- Create a run experiment function that handles creating the environment and running the learning step function provided by the user as a parameter
    - Reports metrics at end following MLFlow Sweeper
    - Accepts hyperparameters in the MLFlow Sweeper format
    - Scans over all steps, jitting the entire training loop
    - Supports batching over multiple seeds, and reports mean and standard deviation of metrics in this case
- LMS sweep, plot hyperparameter sensitivity
- LMS best hyperparameter configuration run and learning curve
- Autostep sweep, plot hyperparameter sensitivity
- Autostep best hyperparameter configuration run and learning curve
- Continual backprop sweep, plot hyperparameter sensitivity
- Continual backprop best hyperparameter configuration run and learning curve
- Fable method sweep, plot hyperparameter sensitivity
- Fable method best hyperparameter configuration run and learning curve
- Learning curves for all methods with best hyperparameter configurations in one plot (data is already collected, so just plot it)

Structure of a sweep cell:
- Define parameters for the sweep and combine with the base config, using 5 seeds per configuration
- Use MLFlow Sweeper's `run_sweep` function with the run experiment function to run the sweep
- Plot the hyperparameter sensitivity curves for each hyperparameter
    - Do not ignore nans, a nan for one configuration
    - If there is a single var, just do a single line plot
    - If there are 2 vars, put one on the x axis and the other as hue
    - If there are 3 vars, make one plot for each, and for each point, take the best performing configuration given that set hyperparameter
- Check which runs ran to convergence algorithmically, and change points that did not converge in some way (like make them smaller or change the alpha or something) to indicate that they did not converge

Structure of a learning curve cell:
- Load the best hyperparameter configuration from the sweep
- Run again the best hyperparameter configuration with 30 seeds
- Plot a learning curve with error bars made using the standard deviation reported

If done in this way with MLFlow Sweeper, the experiments should only need to run in the first execution of the notebook.
Subsequent executions should just load the data from the first execution, as that is what naturally happens when running a sweep with MLFlow Sweeper.