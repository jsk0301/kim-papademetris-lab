# Current Work
My work for Dan's LV masking project is under the lv_masking folder.

- `t1_unet2d.ipynb` contains my own PyTorch implementation of a UNet
- `t2_monai_testing.ipynb` has the things I've tried using the monai package

With access to tensor1, I'm planning to try to come up with a list of different hyperparameters/transformations to test and try training a bunch of models and collect results.

# Roadmap
Now that I have a pretty good idea of what I'm doing, I want to start planning on how to train a model that will actually produce meaningful results.

I'll need more data eventually, so I'm planning to ask Dan to upload everything he has. In the meantime, though, I'll be working off of my smaller current dataset.

Since I'm most familiar with the 2D UNet, I'll start by tweaking/optimizing it as much as possible. Here are some factors I'm thinking of testing:

- Patching vs not patching: this is more of a time/memory saving thing, I'm planning to test both to see how much of a difference this really makes with the models. If patching results in similar accuracy at a much faster rate, I'll stick with it for the rest of my models.
- Data augmentation: I'll be trying out a bunch of different transformations (thanks to MONAI, this should be easy to do) and seeing which ones help.
- Optimizers/Loss
- 

Many output masks seem to be mostly empty - maybe there are too many images with no masking on them?
