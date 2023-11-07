# Inspect Image Models 
To generate image model results: 
- Make sure to change the dicom\_dir and csv\_path in configs files from **./radfusion/configs/dataset**
- Train slice encoder using **run_rsna.sh**. Make sure the download the RSNA RESPECT dataset from [here](https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/rsna-pe-detection-challenge-2020)
- Extract slice representation using **run_featurize.sh**. 
- To train sequence encoder, run hyperparameter search with **wandb sweep sweep.yaml**. Note that line 8 specifies the prediction target. 
