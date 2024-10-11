from pdb_utils import read_sdf, read_pocket_info
from glob import glob
import numpy as np



def find_closest_pocket(pocket_folder, ligand_sdf_file):
    # Use the glob function to obtain the paths of all PDB files (pocket files)
    # in the specified directory, and sort them using np.sort.
    pkt_files = np.sort(glob(f'{pocket_folder}/*.pdb'))

    distance = np.zeros(len(pkt_files))

    pkts_info = []
    # Obtain the center coordinates of the ligand by reading the SDF file.
    lig_center = read_sdf(ligand_sdf_file)[0].GetConformer().GetPositions().mean(axis=0)

    for idx, pkt_file in enumerate(pkt_files):
        # read pocket info, including center, mc_volume, hull_volume
        pkt_info = read_pocket_info(pkt_file)
        pkts_info.append(pkt_info)
        # compute the distance between ligand center and pocket center
        distance[idx] = np.linalg.norm(lig_center - pkt_info[0])

    # find the pocket with the minimum distance
    real_pocket_id = np.argmin(distance)
    # Find the information of the bag with the smallest distance through real_pocket id
    # and obtain its Monte Carlo Volume (MC Volume).
    real_mc_volume = pkts_info[real_pocket_id][1]
    # Similarly, by using real_pocket ID to find the information of the bag with the smallest
    # distance, obtain its Convex Hull Volume.
    real_hull_volume = pkts_info[real_pocket_id][2]

    return real_mc_volume, real_hull_volume


# Example usage:
pocket_folder = './PDB_out/ESR1_3ert_out/pockets'
ligand_sdf_file = './SDF/3ert.sdf'
real_mc_volume, real_hull_volume = find_closest_pocket(pocket_folder, ligand_sdf_file)

print('Monte Carlo Volume: ', real_mc_volume)
print('Convex Hull Volumes: ', real_hull_volume)

