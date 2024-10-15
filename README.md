# MolGenEval: A Benchmark for Comprehensive Evaluation of Molecular Generation Models in De Novo Drug Design
## Overview

De novo molecular design is a computational method that constructs new molecules from scratch, breaking the limitations of traditional virtual screening and enabling the exploration of a broader chemical space for drugs. However, evaluating drug molecule generation models is challenging due to the lack of unified standards, which limits the development of this field. 
In this work, MolGenEval introduces four new evaluation metrics—Time, Available Percentage, Percentage of Active Fragments, and CPMV—based on previous research. These metrics establish a more comprehensive benchmarking system for quantifying the quality of generated molecules and assessing the performance of molecular generation models across five different applications.

## Dependencies

To install all dependencies:
```bash
conda env create -f MolGenEval.yml
```

## Metrics

*  Validity
*  Uniqueness
*  Novelty
*  Available percentage
*  Time
*  Fréchet ChemNet Distance (FCD) 
*  Fragment similarity (Frag)
*  Scaffold similarity (Scaff)
*  Similarity to a nearest neighbor (SNN)
*  Internal diversity (IntDivp) 
*  Percentage of active fragments
*  Molecular docking score
*  Molecular activity prediction
*  Correlation between protein pocket and molecular volume(CPMV)
*  Ring Size
*  Molecular properties distribution
   *  Molecular Weight (MW)
   *  Octanol-Water Partition Coefficient (LogP)
   *  Synthetic Accessibility Score (SA)
   *  Quantitative Estimate of Drug-likeness (QED)
   *  Topological Polar Surface Area (TPSA)
   *  Lipinski's Rule of Five 


## Evaluate molecules

### Usage

Run `./metrics/main.py` program is used for molecular evaluation. You must provide an input file containing SMILES strings for the training set, generation set, active molecules, and inactive molecules. The program calculates metrics to evaluate the generated molecules, prints the results to the console, and outputs the calculated molecular properties to the specified file.

>**Attention**: 
>
>* Please refer to the `CPMV` directory for calculating the CPMV score.
>* Please refer to the `activity_prediction` directory for the calculation of molecular activity prediction scores.

###  Command-Line Arguments

 `--train_set`: The file path for the training dataset (Optional).

`--gen_set`: The file path for the generated dataset (Optional).

 `--output`: The file path where the output will be saved (Optional).

 `--active_set`: The file path for the dataset containing active molecules (Optional).

 `--inactive_set`: The file path for the dataset containing inactive molecules (Optional).

### Example command

```bash

python ./metrics/main.py [--train_set path/to/train.txt] [--gen_set path/to/gen.txt] [--output path/to/output.csv] [--active_set path/to/active.txt] [--inactive_set path/to/inactive.txt]

```

### Output

The results of the metrics are printed to the console and molecular properties and statistics are  saved in a CSV file specified by the `--output` parameter.

### Note

Ensure all input files are formatted correctly with one SMILES string per line. The program utilizes GPU computation if available for performance optimization.
