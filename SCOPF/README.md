Based on [https://github.com/FredericSabot/PDSA-RTS-GMLC](https://github.com/FredericSabot/PDSA-RTS-GMLC), please refer to this source for usage/requirements.

Main file: PSCACOPF.py

Comments:
- NLP problems are difficult to solve numerically, especially if they are not initialized in a good way
- The procedure followed here consists in initializing the final AC PSC-OPF through a succession of preliminary problems
  * A DC PSC-OPF (accounting for an estimation of the losses), to find an estimation of the active power dispatch
  * An AC OPF aiming at finding a feasible AC solution to the power flow equations the closest possible to the solution of the DC PSC-OPF and trying to set the reactive power of generators close to the middle of their capability. The aim is to avoid to push voltages to their upper bound to reduce the losses by generating the maximum of reactive power, at the expense of security (no margin).
  * An AC contingency analysis to find voltages and power flows for all the considered contingency cases
  * Finally, an AC PSCOPF initialized based on the outcome of the previous steps to optimize the convergence properties
    * Contingencies that are not secure in the current dispatch are iteratively added to the optimisation problem for performance reason
    * After a contingency, generators try to keep their terminal voltage equal to the pre-contingency value up to their reactive capabilities
  * Once the current dispatch is statically N-1 secure, dynamic security is checked by simulating double overhead line faults on links along the B4 and B6 boundaries. If unsecure, we add a maximum flow constraint on the problematic boundary and reduce the max value until secure.