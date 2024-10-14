## Introduction

The DRD2. py and JNK3. py programs aim to use pre trained ECFP (Extended Connectivity Fingerprint) classifiers to predict the activity of molecules against DRD2 and JNK3 proteins based on the simplified molecular input line input system (SMILES) representation of molecules. This model utilizes the RDKit library for molecular manipulation and provides an efficient and direct method for activity scoring.

## Usage

The program reads SMILES strings from an input file, calculates their predicted activity scores using a pre-trained model, and writes the results to an output CSV file.

### Command-Line Arguments

- `--input`: Path to the input file containing SMILES strings. Each SMILES should be on a separate line.
- `--output`: Path to the output CSV file where results will be saved.

Example command:

```bash
python DRD2.py/JNK3.py --input path/to/input.txt --output path/to/output.csv
```

### Detailed Workflow

1. **Reading Input**: The script reads SMILES strings from the specified input file.
2. **Model Prediction**: Each SMILES string is converted to a molecular fingerprint, and the pre-trained model predicts the activity score.
3. **Output Results**: The predictions along with the original SMILES strings are saved in a DataFrame and written to the specified output CSV file under the columns "Smiles" and "predit_score".
4. **Completion Message**: Once the processing is complete, a "done!" message is printed to the console.

## Note

Ensure that the input file is properly formatted with one SMILES string per line. The script does not handle empty lines or invalid SMILES strings; such entries should be cleaned or removed before processing.
