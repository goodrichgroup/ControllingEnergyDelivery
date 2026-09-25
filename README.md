# Code for _Controlling energy delivery with bistable nanostructures_

The scripts in this repository perform the numerical calculations shown in Figs. XX of the manuscript [_Controlling energy delivery with bistable nanostructures_](https://doi.org/10.48550/arXiv.2506.14266).

## Instructions
To run the scripts, simply start Python and run the code ``individual_simulation_state.py`` for a Machine-only simulation or ``coupled_simulation_state.py`` for a simulation of a coupled Machine & Source, both at a fixed binding energy $\epsilon$.
This will import all relevant packages and run long Brownian dynamics simulations of the Machine system on its own as well as the coupled system of Machine and Source and save relevant data.
The data contains information about the Machine actions.
For varying binding energies, change the corresponding parameter in the main python script ``coupled_simulation_state.py``.

After that, running the ``Analysis.ipynb`` notebook will help analyzing the data, extracting relevant measures of merit, and producing corresponding figures.


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
