# List of models

5-13-2020:
- Using John's UNet2D script on preliminary small dataset

5-22-2020:
- Created UNet2d from scratch with PyTorch, short training test

5-28-2020:
- Used MONAI implementation of a 2d Unet

6-3-2020:
- Full training test of my from-scratch UNet

6-8-2020:
- Testing effect of patch size: patches were 256x256x1
- Results: mostly messy

6-9-2020:
- Testing effect of patch size: patches were 64x64x1
- Results: no output masks at all??? Theory: blank patches dominate dataset


So a lot of models happened that I didn't update. What I've learned, however, is that the largest factor by FAR is the size of the patch in all three dimensions. I've found that 256x256x16 patches achieve pretty accurate segmentations after just 44 epochs, so this is what I'll go off of.

I need to have better testing metrics than just sampling the masks - I'll have to figure that out.