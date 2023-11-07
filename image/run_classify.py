"""Main pretraining script."""

import hydra


@hydra.main(config_path="./radfusion3/configs", config_name="classify")
def run(config):
    # Deferred imports for faster tab completion
    import os
    import flatten_dict
    import pytorch_lightning as pl
    import radfusion3

    """
    import wandb
    """

    from time import gmtime, strftime
    from omegaconf import OmegaConf
    import re

    pl.seed_everything(config.trainer.seed)

    # Saving checkpoints and logging with wandb.
    flat_config = flatten_dict.flatten(config, reducer="dot")

    # add current time to exp name
    config.exp.name = f"{config.exp.name}_{config.dataset.target}_{strftime('%Y-%m-%d_%H:%M:%S', gmtime())}"
    save_dir = os.path.join(config.exp.base_dir, config.exp.name)
    """
    wandb_logger = pl.loggers.WandbLogger(
        project="impact",
        name=config.exp.name,
        entity="zphuo",
        save_dir="./wandb2",
        log_model="all",
    )
    wandb_logger.log_hyperparams(flat_config)

    # if os.environ.get("LOCAL_RANK", None) is not None:
    # os.environ["WANDB_DIR"] = wandb.run.dir
    global_rank = os.environ.get("JSM_NAMESPACE_RANK")
    if global_rank == 0:
        wandb.define_metric(config.monitor.metric, summary="best", goal="maximize")

        # merge sweep configs
        run = wandb_logger.experiment
        run_config = [f"{k}={v}" for k, v in run.config.items()]
        run_config = OmegaConf.from_dotlist(run_config)
        config = OmegaConf.merge(config, run_config)  # update defaults to CLI
    """
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

    # if .ckpt given then only testing without training
    if config.ckpt is not None and config.ckpt.endswith(".ckpt"):
        ckpt = config.ckpt
        best_epochs = (
            re.search(r"epoch=(\d+).*?-step", ckpt).group(1)
            if re.search(r"epoch=(\d+).*?-step", ckpt)
            else "Not found"
        )
        config.trainer.max_epochs = int(best_epochs)
        print(
            f"best epoch: {config.trainer.max_epochs} ================================"
        )
        ckpt = None
    else:
        ckpt = None
    model = radfusion3.builder.build_lightning_model(config, ckpt=ckpt)
    model.save_dir = save_dir
    # PyTorch Lightning Trainer.
    trainer = pl.Trainer(
        default_root_dir=save_dir,  # '''logger=wandb_logger,'''
        devices=config.n_gpus,
        accelerator="auto",
        strategy=config.trainer.strategy,
        max_epochs=config.trainer.max_epochs,
        val_check_interval=config.trainer.val_check_interval,
        limit_val_batches=config.trainer.limit_val_batches,
        callbacks=callbacks,
        gradient_clip_val=config.trainer.gradient_clip_val,
        precision=config.trainer.precision,
        num_sanity_val_steps=0,
    )

    # find latest ckpt to continue training
    if config.ckpt.endswith("/output") or config.ckpt.endswith("/output/"):
        latest_ckpt = radfusion3.utils.get_latest_ckpt(config)
        trainer.fit(model=model, datamodule=dm, ckpt_path=latest_ckpt)
        trainer.test(datamodule=dm, ckpt_path=latest_ckpt)
    # only testing
    elif config.ckpt.endswith(".ckpt"):
        trainer.fit(model=model, datamodule=dm, ckpt_path=config.ckpt)
        trainer.test(datamodule=dm, ckpt_path=config.ckpt)
    # test withou ckpt
    elif config.ckpt.endswith("test"):
        dm = radfusion3.data.DataModule(config, test_split=config.test_split)
        trainer.test(model=model, datamodule=dm)
    # training from scratch
    else:
        trainer.fit(model=model, datamodule=dm)
        trainer.test(datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    run()
