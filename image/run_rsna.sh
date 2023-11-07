#CUDA_VISIBLE_DEVICES=1 python run_classify.py model=swinv2 dataset=rsna 
#CUDA_VISIBLE_DEVICES=1 python run_classify.py model=dinov2 dataset=rsna dataset.transform.final_size=224 
#python run_classify.py model=dinov2 dataset=rsna dataset.transform.resize_size=512 dataset.transform.crop_size=448 dataset.transform.final_size=448 dataset.batch_size=16
CUDA_VISIBLE_DEVICES=3 python run_classify.py model=resnetv2 dataset=rsna dataset.transform.final_size=224 

#dataset.tranform.final_size=224

