#!/bin/bash
#SBATCH --output=logs/job_%A.out
#SBATCH --error=logs/job_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --mem=100G
#SBATCH --partition=gpu,nigam-a100,nigam-v100
##SBATCH --nodelist=secure-gpu-3,secure-gpu-4,secure-gpu-5,secure-gpu-6,secure-gpu-7,secure-gpu-9,secure-gpu-12
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=20


note_splits=("AnonNote_train_nolabel" "AnonNote_val_nolabel" "AnonNote_test_nolabel")

for note_split in ${note_splits[@]}
do
	python trainer.py --test /share/pi/nigam/projects/zphuo/data/PE/inspect/note/${note_split}.csv \
		--pretrained /share/pi/nigam/projects/mschuang/radnotes/outputs/checkpoint-8590 \
		--pretrained_tokenizer /share/pi/nigam/projects/mschuang/radnotes/Clinical-Longformer \
		--batch_size 8 \
		--max_len 1536 \
		--label_key pe_acute,pe_subsegmentalonly,pe_positive \
		--text_key anon_impression \
		--device cuda:0 \
		--n_epochs 15 \
		--sep ',' \
		--lr 2e-5 \
		--outputdir /share/pi/nigam/projects/zphuo/data/PE/inspect/note
done
