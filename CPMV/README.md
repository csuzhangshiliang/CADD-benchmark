## Introduction
* pdb_utils.py:Some utility classes used in cal_pocket. volume.py.
* cal_pocket_volume.py:To calculate the Monte Carlo Volume and Convex Hull Volume of protein pockets using the find_closest_pocket function, it is necessary to set the PDB file (PDB-out) of the protein and the SDF file (SDF) of the ligand in the pocket.
* cal_CPMV.py:To calculate CPMV (Correlation between protein pocket and molecular volume), it is necessary to input protein volume files and molecular volume files.


## Usage
The `cal_CPMV.py` program requires two input files: one containing protein volumes and another containing molecular volumes. Each file should have numeric data with one volume per line. The program reads these files, computes the Pearson correlation coefficient, and outputs the results.
﻿
## Command-Line Arguments
* `--pro_volume`: The file path for the dataset containing protein volumes (required).
* `--mol_volume`: The file path for the dataset containing molecular volumes (required).

## Example command
```bash
python cal_CPMV.py --pro_volume path/to/protein_volume.txt --mol_volume path/to/molecular_volume.txt
```
## Outputs
* Pearson Correlation Coefficient: Indicates the degree of linear relationship between protein and molecular volumes.
* P-value: Provides the probability score that the observed correlation is due to chance.

## Running the Program
* Prepare your data files with protein volumes and molecular volumes. Ensure each volume is on a new line and the data is clean (numeric values only).
* Use the command line to run the script with the necessary arguments, as shown in the example command.
* Review the printed output for the Pearson correlation coefficient and the P-value.
## Notes
* Ensure that the data files are properly formatted with numeric values. Non-numeric values can cause the program to fail or produce incorrect results.
* The Pearson correlation calculation assumes that the distributions of both datasets are normally distributed. Consider this when interpreting the results.
