import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

def convert_npy_to_hdf5(input_dir, output_path, metadata_path):
    # Read metadata to get mapping between impression_id and image_id
    df_metadata = pd.read_csv(metadata_path)

    # Create HDF5 file
    with h5py.File(output_path, 'w') as hdf5_file:
        # Get all NPY files
        npy_files = list(Path(input_dir).glob('*.npy'))

        for npy_file in npy_files:
            try:
                # Get image_id from filename
                image_id = npy_file.stem

                # Get impression_id from metadata
                impression_id = df_metadata[df_metadata['image_id'] == f"{image_id}.nii.gz"]['impression_id'].iloc[0]

                # Load features
                features = np.load(npy_file)

                # Ensure features are 2D (num_slices x feature_dim)
                if len(features.shape) == 1:
                    features = features.reshape(1, -1)

                # Save to HDF5 using the image_id as key (without .nii.gz)
                hdf5_file.create_dataset(image_id, data=features, dtype='float32')
                print(f"Added {image_id} to HDF5 file with shape {features.shape}")
            except Exception as e:
                print(f"Error processing {npy_file}: {str(e)}")
                continue

if __name__ == "__main__":
    input_dir = "/share/pi/nigam/projects/zphuo/data/PE/inspect/image_modality/anon_pe_features_full_new"
    output_path = "/share/pi/nigam/projects/zphuo/data/PE/inspect/image_modality/anon_pe_features_full_new/features.hdf5"
    metadata_path = "/share/pi/nigam/projects/zphuo/data/PE/inspect/note/Final_metadata.csv"

    convert_npy_to_hdf5(input_dir, output_path, metadata_path)
    print(f"HDF5 file created at: {output_path}")
