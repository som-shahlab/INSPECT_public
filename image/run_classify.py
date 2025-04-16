"""Main pretraining script."""

import hydra


@hydra.main(config_path="./radfusion3/configs", config_name="classify")
def run(config):
    # Deferred imports for faster tab completion
    import os
    import flatten_dict
    import pytorch_lightning as pl
    import radfusion3
    from time import gmtime, strftime
    from omegaconf import OmegaConf
    import re

    pl.seed_everything(config.trainer.seed)

    # Create output directories with proper permissions
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.exp.base_dir, exist_ok=True)

    # Ensure proper permissions (rwx for user and group)
    os.chmod(config.output_dir, 0o770)
    os.chmod(config.exp.base_dir, 0o770)

    # Saving checkpoints and logging with wandb.
    flat_config = flatten_dict.flatten(config, reducer="dot")

    # add current time to exp name
    config.exp.name = f"{config.exp.name}_{config.dataset.target}_{strftime('%Y-%m-%d_%H:%M:%S', gmtime())}"

    save_dir = os.path.join(config.exp.base_dir, config.exp.name)
    os.makedirs(save_dir, exist_ok=True)
    os.chmod(save_dir, 0o770)

    # callbacks
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=save_dir,
        filename="{epoch}-{val/mean_auroc:.3f}",
        save_top_k=1,
        monitor="val/mean_auroc",
        mode="max",
        save_last=True,
    )
    callbacks = [
        lr_monitor,
        checkpoint_callback,
    ]

    # data module
    dm = radfusion3.data.DataModule(config, test_split=config.test_split)

    if config.ckpt and (config.ckpt.endswith("/output") or config.ckpt.endswith("/output/")):
        latest_ckpt = radfusion3.utils.get_latest_ckpt(config)
        config.trainer.max_epochs = int(re.search(r"epoch=(\d+)", latest_ckpt).group(1))
        print(f"Best epoch: {config.trainer.max_epochs}")
        ckpt = None
    else:
        ckpt = None

    model = radfusion3.builder.build_lightning_model(config, ckpt=ckpt)
    model.save_dir = save_dir

    # PyTorch Lightning Trainer
    trainer = pl.Trainer(
        default_root_dir=save_dir,
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

    # Training workflow
    if config.ckpt and config.ckpt.endswith(".ckpt"):
        trainer.fit(model=model, datamodule=dm, ckpt_path=config.ckpt)
        trainer.test(datamodule=dm, ckpt_path="best")
    elif config.ckpt and config.ckpt.endswith("test"):
        trainer.test(model=model, datamodule=dm)
    else:
        trainer.fit(model=model, datamodule=dm)
        trainer.test(datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    run()
