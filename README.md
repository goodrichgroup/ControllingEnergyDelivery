# Code for _Controlling energy delivery with bistable nanostructures_

The scripts in this repository perform the numerical calculations shown in Figs. XX of the manuscript [_Controlling energy delivery with bistable nanostructures_](https://doi.org/10.48550/arXiv.2506.14266).

## Instructions
To run the scripts, simply run the ``slurm_array.sh`` script on your cluster using a Python environment to run an ensemble of long Brownian Dynamics simulations. This will import all relevant packages and run long Brownian Dynamics simulations for an ensemble of Machine-only systems using the code ``individual_simulation_state.py`` or simulation of an ensemble of coupled systems of Machine & Source at a fixed binding energy $\epsilon$ using the code ``coupled_simulation_state.py`` and save relevant data.
The data contains information about the Machine actions and the measures of merit of the energy-delivery mechanism.
For varying binding energies, change the corresponding parameter in the main python script ``coupled_simulation_state.py`` and save data in separate folders for analysis.

After that, running the ``Analysis.ipynb`` notebook will help analyzing the data, extracting relevant measures of merit, and producing corresponding figures.

Comments 
1. All parts of the code that require user input are marked by TODOs.
2. This work is based on the Python libraries [_JAX_](https://github.com/jax-ml/jax) and [_JAX MD_](https://github.com/jax-md/jax-md).

## Citation

```
@article{ehrm26,
  title = {Controlling energy delivery with bistable nanostructures},
  author = {Ehrmann, Andreas and Goodrich, Carl P.},
  year = {2025},
  journal = {arXiv:2506.14266},
  doi = {10.48550/arXiv.2506.14266},
}
```
