# Implemented with PyTorch Lightning
### Imports

# PyTorch imports
import torch
import torch.nn as nn

# MONAI imports
import monai
from monai.transforms import (
    Compose,
    LoadNiftid,
    ScaleIntensityd,
    NormalizeIntensityd,
    AddChanneld,
    ToTensord,
    RandSpatialCropd,
    RandCropByPosNegLabeld,
    CropForegroundd,
    Identityd,
)
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Norm, Act
from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import export
from monai.utils.aliases import alias

# PyTorch Lightning imports
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split

### UNet Model

# Most of the code is copied from MONAI's implenetation of a UNet
class UNet(LightningModule):
    def __init__(
        self,
        data_dir,
        dimensions,
        in_channels,
        out_channels,
        channels,
        strides,
        criterion,
        augmentations,
        kernel_size=3,
        up_kernel_size=3,
        num_res_units=0,
        act=Act.PRELU,
        norm=Norm.INSTANCE,
        dropout=0,
        batch_size=1,
        lr=0.01,
    ):
        super().__init__()
        assert len(channels) == (len(strides) + 1)
        self.data_dir = data_dir
        self.dimensions = dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.criterion = criterion
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.augmentations = augmentations
        self.batch_size = batch_size
        self.lr = lr
        
        self.save_hyperparameters(
            'dimensions',
            'channels',
            'criterion',
            'num_res_units',
            'dropout',
            'augmentations',
            'batch_size',
            'lr'
        )

        def _create_block(inc, outc, channels, strides, is_top):
            """
            Builds the UNet structure from the bottom up by recursing down to the bottom block, then creating sequential
            blocks containing the downsample path, a skip connection around the previous block, and the upsample path.
            """
            c = channels[0]
            s = strides[0]

            if len(channels) > 2:
                subblock = _create_block(
                    c, c, channels[1:], strides[1:], False
                )  # continue recursion down
                upc = c * 2
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self._get_bottom_layer(c, channels[1])
                upc = c + channels[1]

            down = self._get_down_layer(
                inc, c, s, is_top
            )  # create layer in downsampling path
            up = self._get_up_layer(
                upc, outc, s, is_top
            )  # create layer in upsampling path

            return nn.Sequential(down, SkipConnection(subblock), up)

        self.model = _create_block(
            in_channels, out_channels, self.channels, self.strides, True
        )

    def _get_down_layer(self, in_channels, out_channels, strides, is_top):
        if self.num_res_units > 0:
            return ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides,
                self.kernel_size,
                self.num_res_units,
                self.act,
                self.norm,
                self.dropout,
            )
        else:
            return Convolution(
                self.dimensions,
                in_channels,
                out_channels,
                strides,
                self.kernel_size,
                self.act,
                self.norm,
                self.dropout,
            )

    def _get_bottom_layer(self, in_channels, out_channels):
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels, out_channels, strides, is_top):
        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides,
            self.up_kernel_size,
            self.act,
            self.norm,
            self.dropout,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                1,
                self.kernel_size,
                1,
                self.act,
                self.norm,
                self.dropout,
                last_conv_only=is_top,
            )
            return nn.Sequential(conv, ru)
        else:
            return conv

    def forward(self, x):
        x = self.model(x)
        return x


    # Lightning training
    def setup(self, stage):
        data_dir = 'data/'
        
        # Train imgs/masks
        train_imgs = []
        with open(data_dir + 'train_imgs.txt', 'r') as f:
            train_imgs = [image.rstrip() for image in f.readlines()]

        train_masks = []
        with open(data_dir + 'train_masks.txt', 'r') as f:
            train_masks = [mask.rstrip() for mask in f.readlines()]
        
        train_dicts = [{'image': image, 'mask': mask} for (image, mask) in zip(train_imgs, train_masks)]
        
        train_dicts, val_dicts = train_test_split(train_dicts, test_size=0.2)
        
        # Basic transforms
        data_keys = ["image", "mask"]
        data_transforms = Compose(
            [
                LoadNiftid(keys=data_keys),
                AddChanneld(keys=data_keys),
                NormalizeIntensityd(keys="image"),
                RandCropByPosNegLabeld(
                    keys=data_keys, label_key="mask", size=(256, 256, 16), num_samples=4, image_key="image"
                ),
            ]
        )
        
        self.train_dataset = monai.data.CacheDataset(
            data=train_dicts,
            transform=Compose(
                [
                    data_transforms,
                    self.augmentations,
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
                    ToTensord(keys=data_keys)
                ]
            ),
            cache_rate=1.0
        )
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
        
    def train_dataloader(self):
        return monai.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8
        )
    
    def val_dataloader(self):
        return monai.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=8
        )
    
    def training_step(self, batch, batch_idx):
        inputs, labels = (
            batch["image"],
            batch["mask"],
        )
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        tensorboard_logs = {"loss/train": loss}
        return {"loss": loss, "log": tensorboard_logs}

    # Lightning validation
    def validation_step(self, batch, batch_idx):
        inputs, labels = (
            batch["image"],
            batch["mask"],
        )
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"loss/val": avg_loss}
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    
if __name__=="__main__":
    NAME = 'models/6-30-2020/'
    NUM_EPOCHS = 1000

    # criterion = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
    criterion = monai.losses.FocalLoss()

    model = UNet(
        data_dir='data/',
        dimensions=3,
        in_channels=1,
        out_channels=2,
        channels=(64, 128, 258, 512, 1024),
        strides=(2, 2, 2, 2),
        norm=monai.networks.layers.Norm.BATCH,
        criterion=criterion,
        augmentations=Identityd(keys=["image", "mask"]),
        dropout=0,
        num_res_units=2,
	batch_size=2,
    )

    logger = TensorBoardLogger(NAME + "tb_logs/", name='')

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10
    )

    checkpoint_callback = ModelCheckpoint(filepath=NAME + 'checkpoints/')


    trainer = Trainer(
        # auto_lr_find=True,
        # auto_scale_batch_size='binsearch',
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stopping,
        check_val_every_n_epoch=5,
        gpus=1,
        max_epochs=NUM_EPOCHS,
        logger=logger,
    )

    trainer.fit(model)
