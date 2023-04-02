# Comparing the performance of popular charge models for hydration free energy calculations

This repository contains the scripts, MDP files, and some data used in the "Comparing the performance of popular charge models for hydration free energy calculations" (in preparation) manuscript.

Scripts are contained in the `scripts` folder. These scripts presume that you have created a Python environment containing the packages in `environment.yaml`, as well as Psi4 1.6.1. It also assumes the installation of the `charge_model_study` "package" in this repository, with:

```
pip install -e .
```


## Acknowledgements
This project makes use of materials from the Automated Topology Builder (ATB). The ATB is has been and is maintained with support from the University of Queensland (UQ), Australian Research Council (ARC), and Queensland Cyber Infrastructure Foundation (QCIF). To the best of our ability we have not included materials downloaded from the ATB in this repository. In order to access ATB topology and force field files, please visit the [ATB website](https://atb.uq.edu.au/). Please cite the following references if you use materials from this repository pertaining to the ATB:

Malde AK, Zuo L, Breeze M, Stroet M, Poger D, Nair PC, Oostenbrink C, Mark AE.
*An Automated force field Topology Builder (ATB) and repository: version 1.0.*
J. Chem. Theory Comput., **2011**, 7, 4026-4037. DOI:10.1021/ct200196m

Stroet M, Caron B, Visscher K, Geerke D, Malde AK, Mark AE.
*Automated Topology Builder version 3.0: Prediction of solvation free enthalpies in water and hexane.*
J. Chem. Theory Comput. **2018**, 14, 11, 5834-5845 DOI:10.1021/acs.jctc.8b00768

Koziara KB, Stroet M, Malde AK, Mark AE.
*Testing and validation of the Automated Topology Builder (ATB) version 2.0: prediction of hydration free enthalpies.*
J. Comput. Aided. Mol. Des., **2014**, 28, 221-233. DOI:10.1007/s10822-014-9713-7
