from utils import  average_agg_tanimoto, \
   fingerprints,compute_fragments,compute_scaffolds,mapper,get_mol,\
    extract_molecular_fragments,check_substructure,get_MolLogP,get_TPSA,\
    get_QED,get_Lipinski_five,get_SA,get_MolWt,cal_ring_sizes
from scipy.spatial.distance import cosine as cos_distance
import numpy as np
import pandas as pd
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from rdkit import Chem
from fcd_torch import FCD as FCDMetric
from multiprocessing import Pool


import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_valid_smiles(smiles_list: List[str], n_jobs: int = 1) -> List[str]:
    """
    Returns a set of valid SMILES strings from the input set, using the mapper function to handle parallel processing.

    Parameters:
        smiles_list: a list of SMILES strings
        n_jobs: number of threads for calculation

    Returns:
        A set of valid SMILES strings.
    """
    valid_mols = mapper(n_jobs)(get_mol, smiles_list)

    valid_smiles = [smiles for smiles, mol in zip(smiles_list, valid_mols) if mol is not None]

    return valid_smiles


def get_unique_smiles(smiles_list: List[str]) -> List[str]:
    """
    Returns a list of unique SMILES strings from the input list.

    Parameters:
        smiles_list: a list of SMILES strings

    Returns:
        A list of unique SMILES strings, preserving the original order.
    """
    seen = set()
    unique_list = []
    for smiles in smiles_list:
        if smiles not in seen:
            unique_list.append(smiles)
            seen.add(smiles)

    return unique_list


def get_novel_smiles(train_smiles_list: List[str], gen_smiles_list: List[str]) -> List[str]:
    """
    Returns a list of SMILES strings that are present in the generated set but not in the training set.

    Parameters:
        train_smiles_list: a list of SMILES strings in the training set
        gen_smiles_list: a list of SMILES strings in the generated set

    Returns:
        A list of SMILES strings that are only in the generated set and not in the training set.
    """
    train_set = set(train_smiles_list)
    gen_set = set(gen_smiles_list)

    unique_gen_smiles = gen_set - train_set
    return list(unique_gen_smiles)


def get_active_fragments_smiles(active_smiles:List[str],inactive_smiles:List[str],gen:List[str]) -> List[str]:
    '''
    Identifies and returns a list of SMILES strings from a generator that contain unique active molecular fragments not found in the inactive set.

    Parameters:
    active_smiles (List[str]): A list of SMILES strings representing active molecules.
    inactive_smiles (List[str]): A list of SMILES strings representing inactive molecules.
    gen (Iterable[str]): A list of SMILES strings to be checked for containing active fragments.

    Returns:
    List[str]: A list of SMILES strings from 'gen' that contain molecular fragments found only in 'active_smiles'
    and not in 'inactive_smiles'.
    '''
    active_fragments =  extract_molecular_fragments(active_smiles,NumHeavyAtom=8)
    inactive_fragments = extract_molecular_fragments(inactive_smiles, NumHeavyAtom=8)

    active_fragments_set = set(active_fragments)
    print("The number of active molecular fragments:",len(active_fragments_set))
    inactive_fragments_set = set(inactive_fragments)
    print("The number of inactive molecular fragments:",len(inactive_fragments_set))

    unique_active_fragments = active_fragments_set - inactive_fragments_set
    matched_smiles = []
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(check_substructure, smiles, unique_active_fragments): smiles for smiles in gen}
        for future in tqdm(as_completed(futures), total=len(gen), desc="Match SMILES fragments"):
            match = future.result()
            if match:
                matched_smiles.append(match)  # 如果有匹配的smiles，添加到列表中
    print("The number of molecules containing active molecular fragments:",len(matched_smiles))
    return matched_smiles

def calculate_properties(smiles_list: List[str]) -> pd.DataFrame:
    """
    Calculate molecular properties for a list of SMILES strings and return a DataFrame.

    Parameters:
    smiles_list (List[str]): A list of SMILES strings.

    Returns:
    pd.DataFrame: A DataFrame containing the calculated properties for each SMILES string.
    """
    properties = {
        'Smiles': [],
        'MolLogP': [],
        'TPSA': [],
        'QED': [],
        'Lipinski': [],
        'SA': [],
        'MolWt': []
    }

    for smile in tqdm(smiles_list, desc="Generate output file"):
        mol = Chem.MolFromSmiles(smile)
        if mol:  # Check if the molecule was successfully created
            properties['Smiles'].append(smile)
            properties['MolLogP'].append(get_MolLogP(mol))
            properties['TPSA'].append(get_TPSA(mol))
            properties['QED'].append(get_QED(mol))
            properties['Lipinski'].append(get_Lipinski_five(mol))
            properties['SA'].append(get_SA(mol))
            properties['MolWt'].append(get_MolWt(mol))

    return pd.DataFrame(properties)

def calculate_rings_ratio(smiles_list):
    ring_sizes = cal_ring_sizes(smiles_list)
    total_numbers = len(ring_sizes)
    unique_numbers = set(ring_sizes)
    ratio_dict = {}
    for num in unique_numbers:
        count = ring_sizes.count(num)
        ratio = count / total_numbers
        ratio_dict[num] = ratio
    return ratio_dict


'''
Copyright 2018 Insilico Medicine, Inc

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, 
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
def internal_diversity(gen, n_jobs=1, device='cpu', fp_type='morgan',
                       gen_fps=None, p=1):
    """
    Computes internal diversity as:
    1/|A|^2 sum_{x, y in AxA} (1-tanimoto(x, y))
    """
    if gen_fps is None:
        gen_fps = fingerprints(gen, fp_type=fp_type, n_jobs=n_jobs)
    return 1 - (average_agg_tanimoto(gen_fps, gen_fps,
                                     agg='mean', device=device, p=p)).mean()
def compute_intermediate_statistics(smiles, n_jobs=1, device='cpu',
                                    batch_size=512, pool=None):
    """
    The function precomputes statistics such as mean and variance for FCD, etc.
    It is useful to compute the statistics for test and scaffold test sets to
        speedup metrics calculation.
    """
    close_pool = False
    if pool is None:
        if n_jobs != 1:
            pool = Pool(n_jobs)
            close_pool = True
        else:
            pool = 1
    statistics = {}
    mols = mapper(pool)(get_mol, smiles)
    kwargs = {'n_jobs': pool, 'device': device, 'batch_size': batch_size}
    kwargs_fcd = {'n_jobs': n_jobs, 'device': device, 'batch_size': batch_size}
    statistics['FCD'] = FCDMetric(**kwargs_fcd).precalc(smiles)
    statistics['SNN'] = SNNMetric(**kwargs).precalc(mols)
    statistics['Frag'] = FragMetric(**kwargs).precalc(mols)
    statistics['Scaf'] = ScafMetric(**kwargs).precalc(mols)

    if close_pool:
        pool.terminate()
    return statistics
class Metric:
    def __init__(self, n_jobs=1, device='cpu', batch_size=512, **kwargs):
        self.n_jobs = n_jobs
        self.device = device
        self.batch_size = batch_size
        for k, v in kwargs.values():
            setattr(self, k, v)

    def __call__(self, ref=None, gen=None, pref=None, pgen=None):
        assert (ref is None) != (pref is None), "specify ref xor pref"
        assert (gen is None) != (pgen is None), "specify gen xor pgen"
        if pref is None:
            pref = self.precalc(ref)
        if pgen is None:
            pgen = self.precalc(gen)
        return self.metric(pref, pgen)

    def precalc(self, moleclues):
        raise NotImplementedError

    def metric(self, pref, pgen):
        raise NotImplementedError

class SNNMetric(Metric):
    """
    Computes average max similarities of gen SMILES to ref SMILES
    """

    def __init__(self, fp_type='morgan', **kwargs):
        self.fp_type = fp_type
        super().__init__(**kwargs)

    def precalc(self, mols):
        return {'fps': fingerprints(mols, n_jobs=self.n_jobs,
                                    fp_type=self.fp_type)}

    def metric(self, pref, pgen):
        return average_agg_tanimoto(pref['fps'], pgen['fps'],
                                    device=self.device)

def cos_similarity(ref_counts, gen_counts):
    """
    Computes cosine similarity between
     dictionaries of form {name: count}. Non-present
     elements are considered zero:

     sim = <r, g> / ||r|| / ||g||
    """
    if len(ref_counts) == 0 or len(gen_counts) == 0:
        return np.nan
    keys = np.unique(list(ref_counts.keys()) + list(gen_counts.keys()))
    ref_vec = np.array([ref_counts.get(k, 0) for k in keys])
    gen_vec = np.array([gen_counts.get(k, 0) for k in keys])
    return 1 - cos_distance(ref_vec, gen_vec)

class FragMetric(Metric):
    def precalc(self, mols):
        return {'frag': compute_fragments(mols, n_jobs=self.n_jobs)}

    def metric(self, pref, pgen):
        return cos_similarity(pref['frag'], pgen['frag'])

class ScafMetric(Metric):
    def precalc(self, mols):
        return {'scaf': compute_scaffolds(mols, n_jobs=self.n_jobs)}

    def metric(self, pref, pgen):
        return cos_similarity(pref['scaf'], pgen['scaf'])




