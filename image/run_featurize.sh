python run_featurize.py model=resnetv2_ct \
	dataset=stanford \
	dataset.transform.final_size=224 \
	dataset.batch_size=256 \
	dataset.transform.channels=window


python convert_to_hdf5.py