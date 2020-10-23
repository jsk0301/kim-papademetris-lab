import pytorch_lightning as pl
from argparse import Namespace

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightning_modules.UNet3D import UNet3D

if __name__ == '__main__':
    args = {
        'name': '2020-10-14_UNet_NewData',
        'batch_size': 2,
        'lr': 0.001,
        'patch_size': [256, 256, 16],
        'num_workers': 6,
        'data_dir': '../data/new/'
    }

    hparams = Namespace(**args)
    model = UNet3D(hparams)
    
    filepath = '../models/' + hparams.name
    logger = pl.loggers.TensorBoardLogger(filepath + "/tb_logs/", name='')

    # Callbacks
    early_stopping_cb = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5
    )
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        filepath=filepath + '/{epoch:02d}-{val_loss:.2f}'
    )

    trainer = pl.Trainer(
        auto_scale_batch_size='binsearch',
        callbacks=[early_stopping_cb],
        checkpoint_callback=checkpoint_cb,
        check_val_every_n_epoch=5,
        gpus=1,
        max_epochs=1000,
        logger=logger,
    )

    trainer.fit(model)