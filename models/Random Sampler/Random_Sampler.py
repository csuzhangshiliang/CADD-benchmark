from typing import List
import numpy as np


class RandomSampler():
    """
    Generator that samples SMILES strings from a predefined list.
    """

    def __init__(self, molecules: List[str]) -> None:
        """
        Args:
            molecules: list of molecules from which the samples will be drawn
        """
        self.molecules = molecules

    def generate(self, number_samples: int) -> List[str]:
        return list(np.random.choice(self.molecules, size=number_samples))


if __name__ == '__main__':

    smiles_list = []

    with open("train_data.smiles", "r") as file:
        for smiles in file:
            smiles_list.append(smiles.strip())

    randomSampler = RandomSampler(smiles_list)
    generator_smiles = randomSampler.generate(10000)
    print(generator_smiles)
