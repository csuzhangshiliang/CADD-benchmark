import torch
import numpy as np
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as Morgan
import pandas as pd
import scipy.sparse
from rdkit.Chem import rdMMPA
from rdkit.Chem import RDConfig
from tqdm import tqdm

from functools import partial
from multiprocessing import Pool
from rdkit import Chem
from collections import Counter
from rdkit.Chem import AllChem
from typing import Iterable,List
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Descriptors
from rdkit.Chem import QED

import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer


def get_MolLogP(mol):
    return Descriptors.MolLogP(mol)
def get_TPSA(mol):
    return Chem.rdMolDescriptors.CalcTPSA(mol)

def get_QED(mol):
    return QED.qed(mol)

def get_Lipinski_five(mol):
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    h_donors = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)

    score = 0
    score = sum([mw <= 500, logp <= 5, h_donors <= 5, h_acceptors <= 10])
    if score >= 4:
        return 5  # bonus point
    return score

def get_SA(mol):
    try:
        sa = sascorer.calculateScore(mol)
    except:
        sa = -1
    return sa

def get_MolWt(mol):
    return Descriptors.MolWt(mol)

def cal_ring_sizes(smiles_list):
    ring_sizes = []
    for smiles in tqdm(smiles_list,desc="Calculate ring sizes"):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            rings = [len(ring) for ring in mol.GetRingInfo().AtomRings()]
            ring_sizes.extend(rings)
    return ring_sizes

# todo
def mapper(n_jobs):
    '''
    Returns function for map call.
    If n_jobs == 1, will use standard map
    If n_jobs > 1, will use multiprocessing pool
    If n_jobs is a pool object, will return its map function
    '''
    if n_jobs == 1:
        def _mapper(*args, **kwargs):
            return list(map(*args, **kwargs))

        return _mapper
    if isinstance(n_jobs, int):
        pool = Pool(n_jobs)

        def _mapper(*args, **kwargs):
            try:
                result = pool.map(*args, **kwargs)
            finally:
                pool.terminate()
            return result

        return _mapper
    return n_jobs.map

# todo
def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol

def get_mols(smiles_list: Iterable[str]) -> Iterable[Chem.Mol]:
    for i in smiles_list:
        try:
            mol = Chem.MolFromSmiles(i)
            if mol is not None:
                yield mol
        except Exception as e:
            logger.warning(e)

# todo
def fragmenter(mol):
    """
    fragment mol using BRICS and return smiles list
    """
    mol_obj = get_mol(mol)
    if mol_obj is None:
        return []
    fgs = AllChem.FragmentOnBRICSBonds(mol_obj)
    fgs_smi = Chem.MolToSmiles(fgs).split(".")
    return fgs_smi

# todo
def compute_fragments(mol_list, n_jobs=1):
    """
    fragment list of mols using BRICS and return smiles list
    """
    fragments = Counter()
    for mol_frag in mapper(n_jobs)(fragmenter, mol_list):
        fragments.update(mol_frag)
    return fragments


def get_n_rings(mol):
    """
    Computes the number of rings in a molecule
    """
    return mol.GetRingInfo().NumRings()

# todo
def compute_scaffold(mol, min_rings=2):
    mol = get_mol(mol)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except (ValueError, RuntimeError):
        return None
    n_rings = get_n_rings(scaffold)
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    if scaffold_smiles == '' or n_rings < min_rings:
        return None
    return scaffold_smiles

# todo
def compute_scaffolds(mol_list, n_jobs=1, min_rings=2):
    """
    Extracts a scafold from a molecule in a form of a canonic SMILES
    """
    scaffolds = Counter()
    map_ = mapper(n_jobs)
    scaffolds = Counter(
        map_(partial(compute_scaffold, min_rings=min_rings), mol_list))
    if None in scaffolds:
        scaffolds.pop(None)
    return scaffolds

# todo
def average_agg_tanimoto(stock_vecs, gen_vecs,
                         batch_size=5000, agg='max',
                         device='cpu', p=1):
    """
    For each molecule in gen_vecs finds closest molecule in stock_vecs.
    Returns average tanimoto score for between these molecules

    Parameters:
        stock_vecs: numpy array <n_vectors x dim>
        gen_vecs: numpy array <n_vectors' x dim>
        agg: max or mean
        p: power for averaging: (mean x^p)^(1/p)
    """
    assert agg in ['max', 'mean'], "Can aggregate only max or mean"
    agg_tanimoto = np.zeros(len(gen_vecs))
    total = np.zeros(len(gen_vecs))
    for j in range(0, stock_vecs.shape[0], batch_size):
        x_stock = torch.tensor(stock_vecs[j:j + batch_size]).to(device).float()
        for i in range(0, gen_vecs.shape[0], batch_size):
            y_gen = torch.tensor(gen_vecs[i:i + batch_size]).to(device).float()
            y_gen = y_gen.transpose(0, 1)
            tp = torch.mm(x_stock, y_gen)
            jac = (tp / (x_stock.sum(1, keepdim=True) +
                         y_gen.sum(0, keepdim=True) - tp)).cpu().numpy()
            jac[np.isnan(jac)] = 1
            if p != 1:
                jac = jac**p
            if agg == 'max':
                agg_tanimoto[i:i + y_gen.shape[1]] = np.maximum(
                    agg_tanimoto[i:i + y_gen.shape[1]], jac.max(0))
            elif agg == 'mean':
                agg_tanimoto[i:i + y_gen.shape[1]] += jac.sum(0)
                total[i:i + y_gen.shape[1]] += jac.shape[0]
    if agg == 'mean':
        agg_tanimoto /= total
    if p != 1:
        agg_tanimoto = (agg_tanimoto)**(1/p)
    return np.mean(agg_tanimoto)

# todo
def fingerprint(smiles_or_mol, fp_type='maccs', dtype=None, morgan__r=2,
                morgan__n=1024, *args, **kwargs):
    """
    Generates fingerprint for SMILES
    If smiles is invalid, returns None
    Returns numpy array of fingerprint bits

    Parameters:
        smiles: SMILES string
        type: type of fingerprint: [MACCS|morgan]
        dtype: if not None, specifies the dtype of returned array
    """
    fp_type = fp_type.lower()
    molecule = get_mol(smiles_or_mol, *args, **kwargs)
    if molecule is None:
        return None
    if fp_type == 'maccs':
        keys = MACCSkeys.GenMACCSKeys(molecule)
        keys = np.array(keys.GetOnBits())
        fingerprint = np.zeros(166, dtype='uint8')
        if len(keys) != 0:
            fingerprint[keys - 1] = 1  # We drop 0-th key that is always zero
    elif fp_type == 'morgan':
        fingerprint = np.asarray(Morgan(molecule, morgan__r, nBits=morgan__n),
                                 dtype='uint8')
    else:
        raise ValueError("Unknown fingerprint type {}".format(fp_type))
    if dtype is not None:
        fingerprint = fingerprint.astype(dtype)
    return fingerprint

# todo
def fingerprints(smiles_mols_array, n_jobs=1, already_unique=False, *args,
                 **kwargs):
    '''
    Computes fingerprints of smiles np.array/list/pd.Series with n_jobs workers
    e.g.fingerprints(smiles_mols_array, type='morgan', n_jobs=10)
    Inserts np.NaN to rows corresponding to incorrect smiles.
    IMPORTANT: if there is at least one np.NaN, the dtype would be float
    Parameters:
        smiles_mols_array: list/array/pd.Series of smiles or already computed
            RDKit molecules
        n_jobs: number of parralel workers to execute
        already_unique: flag for performance reasons, if smiles array is big
            and already unique. Its value is set to True if smiles_mols_array
            contain RDKit molecules already.
    '''
    if isinstance(smiles_mols_array, pd.Series):
        smiles_mols_array = smiles_mols_array.values
    else:
        smiles_mols_array = np.asarray(smiles_mols_array)
    if not isinstance(smiles_mols_array[0], str):
        already_unique = True

    if not already_unique:
        smiles_mols_array, inv_index = np.unique(smiles_mols_array,
                                                 return_inverse=True)

    fps = mapper(n_jobs)(
        partial(fingerprint, *args, **kwargs), smiles_mols_array
    )

    length = 1
    for fp in fps:
        if fp is not None:
            length = fp.shape[-1]
            first_fp = fp
            break
    fps = [fp if fp is not None else np.array([np.NaN]).repeat(length)[None, :]
           for fp in fps]
    if scipy.sparse.issparse(first_fp):
        fps = scipy.sparse.vstack(fps).tocsr()
    else:
        fps = np.vstack(fps)
    if not already_unique:
        return fps[inv_index]
    return fps

def extract_molecular_fragments(SMILES_list:List[str], NumHeavyAtom:int = 8) -> List[str]:
    '''
    Extracts molecular fragments from a list of SMILES strings that have more than a specified number of heavy atoms.

    Parameters:
    SMILES_list (List[str]): A list of SMILES strings from which to extract molecular fragments.
    NumHeavyAtom (int, optional): The minimum number of heavy atoms a fragment must have to be included in the output. Defaults to 8.

    Returns:
    List[str]: A list of unique SMILES strings representing the fragments that meet the heavy atom count criterion.

    The function processes each SMILES string to generate molecular fragments with specified cuts.
    It filters these fragments to ensure that only those with more than `NumHeavyAtom` heavy atoms are returned.
    Invalid SMILES strings are skipped, and exceptions in fragment generation are handled gracefully.
    '''
    mmps_list = []
    SMILES_list = set(SMILES_list)

    for smile in SMILES_list:
        mol = Chem.MolFromSmiles(smile)
        if not mol:
            continue
        try:
            mmps = rdMMPA.FragmentMol(mol, minCuts=1, maxCuts=1, resultsAsMols=True, maxCutBonds=100,
                                      pattern="[!#1!$(*#*)&!D1]-&!@[!#1!$(*#*)&!D1]")
            for _, fragment in mmps:
                mmps_list.append(fragment)
        except:
            continue

    fragment_smiles = [Chem.MolToSmiles(frag) for frag in mmps_list if frag]
    frag_list = []
    for smiles in fragment_smiles:
        frag_list.extend(smiles.split('.'))

    final_list = []
    for frag_smile in frag_list:
        mol = Chem.MolFromSmiles(frag_smile)
        if mol and mol.GetNumHeavyAtoms() > NumHeavyAtom:
            final_list.append(Chem.MolToSmiles(mol))

    return list(set(final_list))

def is_substructure(substructure, molecule_smiles):
    '''
    Determines whether a given substructure is present within a specified molecule represented by a SMILES string.

    Parameters:
    substructure (str): The string representing the substructure to search for.
    molecule_smiles (str): The SMILES string representing the molecule in which to search for the substructure.

    Returns:
    bool: Returns True if the substructure is found in the molecule; otherwise, returns False.
    '''
    substructure = Chem.MolFromSmarts(substructure)
    molecule = Chem.MolFromSmiles(molecule_smiles)

    # 检查分子对象是否成功创建
    if substructure is None or molecule is None:
        raise ValueError("Invalid input")


    matches = molecule.GetSubstructMatches(substructure)
    if matches:
        return True
    else:
        return False

def check_substructure(smiles, unique_active_fragments):
    for fragments in unique_active_fragments:
        if is_substructure(fragments, smiles):
            return smiles
    return None
