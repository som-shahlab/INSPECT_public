"""Main pretraining script."""

import hydra


@hydra.main(config_path="./radfusion3/configs", config_name="classify")
def run(config):
    # Deferred imports for faster tab completion
    import os
    import flatten_dict
    import pytorch_lightning as pl
    import radfusion3
    import wandb

    from time import gmtime, strftime
    from omegaconf import OmegaConf

    pl.seed_everything(config.trainer.seed)

    # Saving checkpoints and logging with wandb.
    flat_config = flatten_dict.flatten(config, reducer="dot")

    # add current time to exp name
    config.exp.name = f"{config.exp.name}_{config.dataset.target}_{strftime('%Y-%m-%d_%H:%M:%S', gmtime())}"
    save_dir = os.path.join(config.exp.base_dir, config.exp.name)
    wandb_logger = pl.loggers.WandbLogger(
        project="impact",
        name=config.exp.name,
        entity="zphuo",
        save_dir="./wandb2",
        log_model="all",
    )
    wandb_logger.log_hyperparams(flat_config)
    wandb.define_metric(config.monitor.metric, summary="best", goal="maximize")

    # merge sweep configs
    run = wandb_logger.experiment
    run_config = [f"{k}={v}" for k, v in run.config.items()]
    run_config = OmegaConf.from_dotlist(run_config)
    config = OmegaConf.merge(config, run_config)  # update defaults to CLI

    # call backs
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=save_dir,
        every_n_epochs=1,
        save_top_k=-1,
        monitor=config.monitor.metric,
        mode=config.monitor.mode,
    )
    callbacks = [
        lr_monitor,
        checkpoint_callback,
    ]

    # data module
    dm = radfusion3.data.DataModule(config, test_split=config.test_split)

    model = radfusion3.builder.build_lightning_model(config)
    model.save_dir = save_dir
    # PyTorch Lightning Trainer.
    trainer = pl.Trainer(
        default_root_dir=save_dir,
        logger=wandb_logger,
        devices=config.n_gpus,
        accelerator="auto",
        max_epochs=config.trainer.max_epochs,
        val_check_interval=config.trainer.val_check_interval,
        limit_val_batches=config.trainer.limit_val_batches,
        callbacks=callbacks,
        gradient_clip_val=config.trainer.gradient_clip_val,
        precision=config.trainer.precision,
        num_sanity_val_steps=0,
    )
    trainer.fit(model=model, datamodule=dm)
    trainer.test(datamodule=dm)


if __name__ == "__main__":
    run()
