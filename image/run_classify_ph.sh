#!/bin/bash
#SBATCH --job-name=ph
#SBATCH --output=logs/job_%A_out.log
#SBATCH --error=logs/job_%A_err.log
#SBATCH --time=2-00:00:00
#SBATCH --mem=100G
#SBATCH --nodes=1 
#SBATCH --partition=nigam-a100
##SBATCH --nodelist=secure-gpu-9,secure-gpu-12,secure-gpu-3,secure-gpu-4,secure-gpu-5,secure-gpu-6,secure-gpu-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10

# seeds="0 1 2 3 4 5 6 7 8 9"
seeds="0"
n_gpus=1
strategy=ddp
batch_size=128
num_workers=4
num_slices=250

echo "***************" "12_month_PH" "********************************************"
for seed in $seeds
do
python run_classify.py model=model_1d dataset=stanford_featurized \
	dataset.target=12_month_PH\
	dataset.pretrain_args.model_type=resnetv2_101_ct\
	dataset.pretrain_args.channel_type=window\
	dataset.feature_size=768 \
	dataset.num_slices=250 \
	model.aggregation=attention \
	model.seq_encoder.rnn_type=GRU \
	model.seq_encoder.bidirectional=true\
	model.seq_encoder.num_layers=1\
	model.seq_encoder.hidden_size=128\
	model.seq_encoder.dropout_prob=0.25\
	dataset.weighted_sample=true \
	trainer.max_epochs=50\
	lr=0.001 \
	trainer.seed=$seed \
	n_gpus=$n_gpus\
	trainer.strategy=${strategy}\
	dataset.batch_size=${batch_size} \
	trainer.num_workers=${num_workers} \
	dataset.num_slices=${num_slices} 
done