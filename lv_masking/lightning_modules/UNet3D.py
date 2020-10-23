import numpy as np

import monai
from monai.transforms import(
    Compose,
    LoadNiftid,
    AddChanneld,
    ScaleIntensityRangePercentilesd,
    RandCropByPosNegLabeld,
    CenterSpatialCropd,
    ToTensord
)

import pytorch_lightning as pl

import torch
import torchvision

from sklearn.model_selection import train_test_split

class UNet3D(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.unet = monai.networks.nets.UNet(
            dimensions=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            norm=monai.networks.layers.Norm.BATCH,
            num_res_units=2
        )
        self.sample_masks = []
    
    def prepare_data(self):
        data_dir = self.hparams.data_dir
        
        # Train imgs/masks
        train_imgs = []
        train_masks = []
        with open(data_dir + 'train_imgs.txt', 'r') as f:
            train_imgs = [data_dir + image.rstrip() for image in f.readlines()]
        with open(data_dir + 'train_masks.txt', 'r') as f:
            train_masks = [data_dir + mask.rstrip() for mask in f.readlines()]
        train_dicts = [{'image': image, 'mask': mask} for (image, mask) in zip(train_imgs, train_masks)]
        train_dicts, val_dicts = train_test_split(train_dicts, test_size=0.2)
        
        # Basic transforms
        data_keys = ["image", "mask"]
        data_transforms = Compose(
            [
                LoadNiftid(keys=data_keys),
                AddChanneld(keys=data_keys),
                ScaleIntensityRangePercentilesd(
                    keys='image',
                    lower=25,
                    upper=75,
                    b_min=-0.5,
                    b_max=0.5
                )
            ]
        )
        
        self.train_dataset = monai.data.CacheDataset(
            data=train_dicts,
            transform=Compose(
                [
                    data_transforms,
                    RandCropByPosNegLabeld(
                        keys=data_keys,
                        label_key="mask",
                        spatial_size=self.hparams.patch_size,
                        num_samples=4,
                        image_key="image",
                        pos=0.8,
                        neg=0.2
                    ),
                    ToTensord(keys=data_keys)
                ]
            ),
            cache_rate=1.0
        )
        
        self.val_dataset = monai.data.CacheDataset(
            data=val_dicts,
            transform=Compose(
                [
                    data_transforms,
                    CenterSpatialCropd(keys=data_keys, roi_size=self.hparams.patch_size),
                    ToTensord(keys=data_keys)
                ]
            ),
            cache_rate=1.0
        )
        
    def train_dataloader(self):
        return monai.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers
        )

    def val_dataloader(self):
        return monai.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers
        )
    
    # Training setup
    def forward(self, image):
        return self.unet(image)
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch['image'], batch['mask']
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.logger.log_metrics({"loss/train": loss}, self.global_step)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = (
            batch["image"],
            batch["mask"],
        )
        outputs = self(inputs)
        # Sample masks
        if self.current_epoch != 0:
            image = outputs[0].argmax(0)[:, :, 8].unsqueeze(0).detach()
            self.sample_masks.append(image)
        loss = self.criterion(outputs, labels)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.logger.log_metrics({"val/loss": avg_loss}, self.current_epoch)
        if self.current_epoch != 0:
            grid = torchvision.utils.make_grid(self.sample_masks)
            self.logger.experiment.add_image('sample_masks', grid, self.current_epoch)
            self.sample_masks = []
        return {"val_loss": avg_loss}
    
    def criterion(self, y_hat, y):
        dice_loss = monai.losses.DiceLoss(
            to_onehot_y=True,
            softmax=True
        )
        focal_loss = monai.losses.FocalLoss()
        return dice_loss(y_hat, y) + focal_loss(y_hat, y)

    def configure_optimizers(self):
        lr = self.hparams.lr
        optimizer = torch.optim.Adam(self.unet.parameters(), lr=lr)
        return optimizer