# Implemented with PyTorch Lightning
### Imports
import os
import glob
import argparse

# PyTorch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
)

# PyTorch Lightning imports
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

# UNet model
from MONAIUNet import UNet

### Helper functions
def select_animals(images, masks, animals):
    """Returns the images and masks as a dictionary for specific animals."""
    filtered_images = []
    filtered_masks = []
    for animal in animals:
        filtered_images.extend(filter(lambda x: "PSEA" + str(animal) in x, images))
        filtered_masks.extend(filter(lambda x: "PSEA" + str(animal) in x, masks))
    return [
        {"image": image_file, "mask": mask_file}
        for image_file, mask_file in zip(filtered_images, filtered_masks)
    ]


def create_loader(
    data_dicts, transforms=None, batch_size=1, shuffle=False, num_workers=8
):
    """Creates a MONAI CacheDataset from data dictionaries."""
    dataset = monai.data.CacheDataset(
        data=data_dicts, transform=transforms, cache_rate=1.0
    )
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return data_loader


def get_data_loaders(batch_size, patch_size):
    ### Data collection
    data_dir = "data/"
    print("Available directories: ", os.listdir(data_dir))
    # Get paths for images and masks, organize into dictionaries
    images = sorted(glob.glob(data_dir + "**/*CTImg*", recursive=True))
    masks = sorted(glob.glob(data_dir + "**/*Mask*", recursive=True))
    # Dataset selection
    train_dicts = select_animals(images, masks, [12, 13, 14, 18, 20])
    val_dicts = select_animals(images, masks, [25])
    data_keys = ["image", "mask"]
    # Data transformation
    data_transforms = Compose(
        [
            LoadNiftid(keys=data_keys),
            AddChanneld(keys=data_keys),
            NormalizeIntensityd(keys=data_keys),
            RandCropByPosNegLabeld(
                keys=data_keys, label_key="mask", size=(patch_size, patch_size, 1), num_samples=30
            ),
            ToTensord(keys=data_keys),
        ]
    )

    # Data loaders
    data_loaders = {
        "train": create_loader(
            train_dicts,
            batch_size=batch_size,
            transforms=data_transforms,
            shuffle=True,
        ),
        "val": create_loader(val_dicts, transforms=data_transforms),
    }
    for key in data_loaders:
        print(key, len(data_loaders[key]))

    return data_loaders


### Main
def main(hparams):
    print("===== INITIAL PARAMETERS =====")
    print("Model name: ", hparams.name)
    print("Batch size: ", hparams.batch_size)
    print("Patch size: ", hparams.patch_size)
    print("Epochs: ", hparams.epochs)
    print("Learning rate: ", hparams.learning_rate)
    print()

    data_loaders = get_data_loaders(hparams.batch_size, hparams.patch_size)

    ### Model training
    criterion = monai.losses.DiceLoss(to_onehot_y=True, do_softmax=True)

    model = UNet(
        dimensions=2,
        in_channels=1,
        out_channels=2,
        channels=(64, 128, 258, 512, 1024),
        strides=(2, 2, 2, 2),
        norm=monai.networks.layers.Norm.BATCH,
        criterion=criterion,
        hparams=hparams,
    )
    logger = TensorBoardLogger("tb_logs", name=hparams.name)

    trainer = Trainer(
        check_val_every_n_epoch=5,
        default_save_path=hparams.name + "/checkpoints",
        gpus=1,
        max_epochs=hparams.epochs,
        logger=logger,
    )

    trainer.fit(
        model,
        train_dataloader=data_loaders["train"],
        val_dataloaders=data_loaders["val"],
    )


### Argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a model based on MONAI's UNet.")
    parser.add_argument("--name", type=str, required=True, help="Name of the model.")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size. Default: 64"
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Width and height of patches. Default: 256.",
    )
    parser.add_argument(
        "--epochs", type=int, default=1000, help="Number of epochs. Default: 1000."
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Initial learning rate. Default: 0.001.",
    )
    args = parser.parse_args()

    main(args)
