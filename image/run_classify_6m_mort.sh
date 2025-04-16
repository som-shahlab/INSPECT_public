#!/bin/bash
#SBATCH --job-name=6m_mort
#SBATCH --output=logs/job_%A_out.log
#SBATCH --error=logs/job_%A_err.log
#SBATCH --time=2-00:00:00
#SBATCH --mem=200G
#SBATCH --nodes=1 
#SBATCH --partition=nigam-v100
##SBATCH --nodelist=secure-gpu-9,secure-gpu-12,secure-gpu-3,secure-gpu-4,secure-gpu-5,secure-gpu-6,secure-gpu-7
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=5

# seeds="0 1 2 3 4 5 6 7 8 9"
seeds="0"
n_gpus=2
strategy=ddp
batch_size=128
num_workers=4
num_slices=250

echo "***************" "6_month_mortality" "********************************************"
for seed in $seeds
do
python run_classify.py model=model_1d dataset=stanford_featurized \
	dataset.target=6_month_mortality\
	dataset.pretrain_args.model_type=resnetv2_101_ct\
	dataset.pretrain_args.channel_type=window\
	dataset.feature_size=768 \
	dataset.num_slices=250 \
	model.aggregation=mean\
	model.seq_encoder.rnn_type=GRU \
	model.seq_encoder.bidirectional=true\
	model.seq_encoder.num_layers=1\
	model.seq_encoder.hidden_size=128\
	model.seq_encoder.dropout_prob=0.0\
	dataset.weighted_sample=true\
	trainer.max_epochs=50\
	trainer.val_check_interval=1.0\
	trainer.limit_val_batches=1.0\
	lr=0.0005 \
	trainer.seed=$seed \
	n_gpus=$n_gpus\
	trainer.strategy=${strategy}\
	dataset.batch_size=${batch_size} \
	trainer.num_workers=${num_workers} \
	dataset.num_slices=${num_slices} 
done
