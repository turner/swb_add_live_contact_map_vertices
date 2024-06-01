import argparse
import numpy as np
import h5py

def filter_by_prefix(strings, prefix):
    return [string for string in strings if string.startswith(prefix)]

def numerical_sort_key(s):
    return int(s.split('_')[1])

def find_group(hdf, group_name, current_path='/'):
    """
    Recursively search for a group with the given name in the HDF5 file.
    
    :param hdf: HDF5 file object
    :param group_name: Name of the group to search for
    :param current_path: Current path in the HDF5 file (used for recursion)
    :return: Path to the group if found, otherwise None
    """
    if group_name in hdf[current_path].keys():
        return current_path + group_name
    else:
        for key in hdf[current_path].keys():
            item = hdf[current_path + key]
            if isinstance(item, h5py.Group):
                result = find_group(hdf, group_name, current_path + key + '/')
                if result is not None:
                    return result
    return None

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Search for a group in an HDF5 file.')
    parser.add_argument('-f', '--filename', type=str, required=True, help='HDF5 filename')
    parser.add_argument('-p', '--prefix', type=str, required=True, help='Prefix shared amongst all ensemble groups')

    args = parser.parse_args()

    # Open the HDF5 file in read-write mode
    with h5py.File(args.filename, 'r+') as hdf:

        keys = hdf.keys()
        parent_group_names = filter_by_prefix(keys, args.prefix)

        for parent_group_name in parent_group_names:

            parent_path = find_group(hdf, parent_group_name)
            parent_group = hdf[parent_path]

            group_path = find_group(hdf, 'spatial_position')
            group = hdf[group_path]

            datasets = []
            dataset_names = sorted(group.keys(), key=numerical_sort_key)
            for dataset_name in dataset_names:
                dataset = group[dataset_name]
                datasets.append(dataset[:])

            # Concatenate all datasets along the first axis
            concatenated_data = np.concatenate(datasets, axis=0)

            # Create a new dataset for the concatenated data
            lcmv = 'live_contact_map_vertices'
            if lcmv in parent_group:
                del parent_group[lcmv]  # Delete existing dataset if it exists

            parent_group.create_dataset(lcmv, data=concatenated_data)

            print(f"New dataset '{lcmv}' created with shape: {concatenated_data.shape}")


if __name__ == "__main__":
    main()
