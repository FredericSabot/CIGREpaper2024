This repository contains the data used for the paper "Impact of active distribution networks on power system stability - a case study" presented at the CIGRE Paris Session 2024.

# Organisation

The directory structure of the repository is organised as follows:

- FES data contains regional breakdown of the estimated distributed energy source installed capacities as predicted by the Future Energy Scenarios
- Powerfactory contains the transmission system model associated with dynamic equivalents of the distribution networks. The network dispatches for all considered scenarios are also included.
- SCOPF contains the code used to dispatch the transmission grid model.
- distrib_networks contains the distribution grid models and the code used to generate their dynamic equivalents.


# Citation

If you find this code useful, please cite the following paper:

```
@INPROCEEDINGS{CIGRE_paper,
  author={Sabot, Frédéric and Henneaux, Pierre and Lamprianidou, Ifigeneia S. and Papadopoulos, Panagiotis N. and Bell, Keith},
  booktitle={CIGRE Paris Session 2024},
  title={Impact of active distribution networks on power system stability - a case study},
  year={2024},
  month = 8,
}
```

If you use the representative model of Scotland and Northern England transmission grid, please cite and refer to Samuel Gordon's PhD thesis.

```
@phdthesis{Sam_thesis,
	title = {Dynamic Interactions Between Voltage and Frequency Events in Future Power Systems},
	school = {University of Strathclyde},
	author = {Gordon, Samuel},
	year = {2024},
}
```