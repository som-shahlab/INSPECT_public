python3 trainer.py --train /share/pi/nigam/projects/mschuang/radnotes/imon_cohort_anon/train.csv \
	--valid /share/pi/nigam/projects/mschuang/radnotes/imon_cohort_anon/val.csv \
	--pretrained ./Clinical-Longformer \
	--pretrained_tokenizer ./Clinical-Longformer \
	--batch_size 4 \
	--max_len 1536 \
	--label_key pe_acute,pe_subsegmentalonly,pe_positive \
	--text_key anon_impression \
	--device cuda \
	--device_ids 0 \
	--sep ',' \
	--n_epochs 15 \
	--lr 2e-5 \
	--outputdir /share/pi/nigam/projects/mschuang/radnotes/outputs

	#--batch_size 8 \

