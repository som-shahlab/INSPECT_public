import pytorch_lightning as pl
import random
from torch.utils.data import DataLoader, SubsetRandomSampler
from .. import builder


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg, test_split="test"):
        super().__init__()

        self.cfg = cfg
        self.dataset = builder.build_dataset(cfg)
        self.test_split = test_split

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        # Split train into train/val if no validation split exists
        if not hasattr(self, '_using_train_valid_split'):
            self._using_train_valid_split = True
            # Get all indices
            all_indices = list(range(len(dataset)))
            # Use 10% for validation
            val_size = int(0.1 * len(dataset))
            # Use fixed seed for reproducibility
            random.seed(42)
            self._val_indices = random.sample(all_indices, val_size)
            self._train_indices = list(set(all_indices) - set(self._val_indices))
            
        if self.cfg.dataset.weighted_sample:
            sampler = dataset.get_sampler(indices=self._train_indices)
            return DataLoader(
                dataset,
                pin_memory=True,
                drop_last=True,
                shuffle=False,
                sampler=sampler,
                batch_size=self.cfg.dataset.batch_size,
                num_workers=self.cfg.trainer.num_workers,
            )
        else:
            sampler = SubsetRandomSampler(self._train_indices)
            return DataLoader(
                dataset,
                pin_memory=True,
                drop_last=True,
                sampler=sampler,
                batch_size=self.cfg.dataset.batch_size,
                num_workers=self.cfg.trainer.num_workers,
            )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "val")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        # Use validation indices if we created them
        if hasattr(self, '_val_indices'):
            sampler = SubsetRandomSampler(self._val_indices)
            return DataLoader(
                dataset,
                pin_memory=True,
                drop_last=False,
                sampler=sampler,
                batch_size=self.cfg.dataset.batch_size,
                num_workers=self.cfg.trainer.num_workers,
            )
        else:
            # Try to load validation split
            dataset = self.dataset(self.cfg, split="valid", transform=transform)
            return DataLoader(
                dataset,
                pin_memory=True,
                drop_last=False,
                shuffle=False,
                batch_size=self.cfg.dataset.batch_size,
                num_workers=self.cfg.trainer.num_workers,
            )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, self.test_split)
        dataset = self.dataset(self.cfg, split=self.test_split, transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            batch_size=self.cfg.dataset.batch_size,
            num_workers=self.cfg.trainer.num_workers,
        )

    def all_dataloader(self):
        transform = builder.build_transformation(self.cfg, "all")
        dataset = self.dataset(self.cfg, split=self.cfg.test_split, transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            batch_size=self.cfg.dataset.batch_size,
            num_workers=self.cfg.trainer.num_workers,
        )
