"""Main pretraining script."""

import hydra


@hydra.main(config_path="./radfusion3/configs", config_name="extract")
def run(config):
    # Deferred imports for faster tab completion
    import os
    import flatten_dict
    import pytorch_lightning as pl
    import radfusion3

    from time import gmtime, strftime

    pl.seed_everything(config.trainer.seed)

    # Saving checkpoints and logging with wandb.
    flat_config = flatten_dict.flatten(config, reducer="dot")
    # add current time to exp name
    config.exp.name = f"{config.exp.name}_{strftime('%Y-%m-%d_%H:%M:%S', gmtime())}"
    save_dir = os.path.join(config.exp.base_dir, config.exp.name)
    """
    wandb_logger = pl.loggers.WandbLogger(project="radfusion3", name=config.exp.name)
    wandb_logger.log_hyperparams(flat_config)
    """

    # call backs
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=save_dir,
        filename='model-{epoch:02d}',
        monitor='train/mean_auroc',  # Using training metrics since validation metrics aren't available
        mode='max',
        save_top_k=1,
        save_last=True,
        every_n_epochs=1
    )
    callbacks = [
        lr_monitor,
        checkpoint_callback,
    ]

    model = radfusion3.builder.build_lightning_model(config)
    dm = radfusion3.data.DataModule(config, test_split=config.test_split)

    # PyTorch Lightning Trainer.
    trainer = pl.Trainer(
        default_root_dir=save_dir,
        # '''logger=wandb_logger,'''
        devices=config.n_gpus,
        accelerator="auto",
        max_steps=config.trainer.max_steps,
        min_steps=config.trainer.max_steps,
        # Using limit_val_batches=0.0 will disable validation entirely
        limit_val_batches=0.0,
        callbacks=callbacks,
        gradient_clip_val=config.trainer.gradient_clip_val,
        precision=config.trainer.precision,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        max_epochs=config.trainer.max_epochs,
    )

    # trainer.test(model=model, datamodule=dm)
    print(len(dm.all_dataloader()), "++++++++++++++++++++++++++++++++++++++++++++++++")
    print(
        len(dm.train_dataloader()), "++++++++++++++++++++++++++++++++++++++++++++++++"
    )
    print(len(dm.val_dataloader()), "++++++++++++++++++++++++++++++++++++++++++++++++")
    print(len(dm.test_dataloader()), "++++++++++++++++++++++++++++++++++++++++++++++++")

    # First fit the model
    trainer.fit(model=model, datamodule=dm)

    # Then test using the last checkpoint instead of best
    trainer.test(datamodule=dm, ckpt_path="last")


if __name__ == "__main__":
    run()
