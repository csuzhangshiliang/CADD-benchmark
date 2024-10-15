import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from fcd_torch import FCD as FCDMetric
from metrics import get_valid_smiles, get_unique_smiles, get_novel_smiles, \
    SNNMetric, ScafMetric, FragMetric, internal_diversity,calculate_properties,\
    compute_intermediate_statistics,get_active_fragments_smiles,calculate_rings_ratio
import argparse
from utils import get_mol, mapper


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_set', type=str, required=False, default='../data/example_train_set.smiles',help='train_set_file')
    parser.add_argument('--gen_set', type=str, required=False, default='../data/example_gen_set.smiles',help='gen_set_file')
    parser.add_argument('--output', type=str, required=False, default='../data/output.csv',help='output_file_path')
    parser.add_argument('--active_set', type=str, required=False, default='../data/activate_data/DRD2-activate.txt',help='active_molecule_set')
    parser.add_argument('--inactive_set', type=str, required=False, default='../data/example_inactive_set.smiles',help='inactive_molecule_set')

    args = parser.parse_args()
    train_file = args.train_set
    gen_file = args.gen_set
    output_path = args.output
    active_file = args.active_set
    inactive_file = args.inactive_set

    training_smiles = []
    gen_smiles = []
    active_smiles = []
    inactive_smiles = []
    with open(train_file, 'r') as train_list:
        for line in train_list:
            smiles = line.strip()
            training_smiles.append(smiles)
    with open(gen_file, 'r') as gen_list:
        for line in gen_list:
            smiles = line.strip()
            gen_smiles.append(smiles)
    with open(active_file, 'r') as active_list:
        for line in active_list:
            smiles = line.strip()
            active_smiles.append(smiles)
    with open(inactive_file, 'r') as inactive_list:
        for line in inactive_list:
            smiles = line.strip()
            inactive_smiles.append(smiles)



    metric = {}
    valid_smiles = get_valid_smiles(gen_smiles)
    metric["validity"] = len(valid_smiles) / len(gen_smiles)
    unique_smiles = get_unique_smiles(valid_smiles)
    metric["uniqueness"] = len(unique_smiles) / len(valid_smiles)
    novel_smiles = get_novel_smiles(training_smiles, unique_smiles)
    metric["novelty"] = len(novel_smiles) / len(unique_smiles)
    metric["available_percentage"] = metric["validity"] * metric["uniqueness"] * metric["novelty"]

    active_fragments_smiles = get_active_fragments_smiles(active_smiles,inactive_smiles,valid_smiles)
    metric["percentage_of_active_fragments"] = len(active_fragments_smiles)/len(valid_smiles)


    ptest = compute_intermediate_statistics(valid_smiles, n_jobs=8,
                                            device='cuda',
                                            batch_size=512)

    mols = mapper(n_jobs=8)(get_mol, valid_smiles)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric["FCD"] = FCDMetric(n_jobs=8, device=device, batch_size=512)(gen=valid_smiles, pref=ptest['FCD'])
    metric["SNN"] = SNNMetric(n_jobs=8, device=device, batch_size=512)(gen=mols, pref=ptest['SNN'])
    metric["Frag"] = FragMetric(n_jobs=8, device=device, batch_size=512)(gen=mols, pref=ptest['Frag'])
    metric["Scaff"]= ScafMetric(n_jobs=8, device=device, batch_size=512)(gen=mols, pref=ptest['Scaf'])
    metric["internal_diversity"] = internal_diversity(valid_smiles, n_jobs=8, device=device)
    print("=========================Calculation results==================================")
    for k, v in metric.items():
        print(f"{k}: {v:.2%}")

    train_ratio_dict = calculate_rings_ratio(training_smiles)
    for num, ratio in train_ratio_dict .items():
        if num in range(3, 10):
            print(f"train_set_ring_sizes {num}: {ratio:.2%}")

    valid_ratio_dict = calculate_rings_ratio(valid_smiles)
    for num, ratio in valid_ratio_dict .items():
        if num in range(3, 10):
            print(f"gen_set_ring_sizes {num}: {ratio:.2%}")

    result = calculate_properties(valid_smiles)
    result.to_csv(output_path)



