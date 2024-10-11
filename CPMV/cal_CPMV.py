import argparse
from scipy import stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pro_volume', type=str, required=True, help='protein_volume_file')
    parser.add_argument('--mol_volume', type=str, required=True, help='molecular_volume_file')

    args = parser.parse_args()
    pro_volume_file = args.pro_volume
    mol_volume_file = args.mol_volume

    pro_volume = []
    mol_volume = []
    with open(pro_volume_file, 'r', encoding='utf-8') as file:
        for lines in file:
            if lines.strip():
                pro_volume.append(lines.strip())

    with open(mol_volume_file, 'r', encoding='utf-8') as file:
        for lines in file:
            if lines.strip():
                mol_volume.append(lines.strip())


    # Calculate Pearson correlation coefficient
    correlation_coefficient, p_value = stats.pearsonr(pro_volume, mol_volume)
    print("Pearson Correlation Coefficient:", correlation_coefficient)
    print("P-value:", p_value)



