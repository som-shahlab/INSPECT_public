python run_classify.py model=model_1d dataset=stanford_featurized \
	dataset.target=pe_positive_nlp \
	dataset.pretrain_args.model_type=resnetv2_101_ct\
	dataset.pretrain_args.channel_type=window\
	dataset.feature_size=768 \
	dataset.num_slices=250 \
	model.aggregation=attention \
	model.seq_encoder.rnn_type=GRU \
	model.seq_encoder.bidirectional=true\
	model.seq_encoder.num_layers=1\
	model.seq_encoder.dropout_prob=0.5\
	dataset.weighted_sample=false \
	trainer.max_epochs=1\
	lr=0.001 \
