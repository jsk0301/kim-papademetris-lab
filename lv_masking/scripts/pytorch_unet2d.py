import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import nibabel as nib

NUM_EPOCHS = 100
LEARNING_RATE = 0.0001
TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
datapath = 'data/'

# Data
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class LVDataset(Dataset):
    # animal_nums is a list of numbers indicating which animals' images to include in the dataset
    
    # to try: tuple(zip(images, masks))
    def __init__(self, animal_nums, transform=None):
        image_folders = []
        types = ['Baseline', 'PostGel', 'PostMI']
        for num in animal_nums:
            for type in types:
                image_folders.append(datapath + 'PSEA' + str(num) + ' ' + type + '/')
        self.image_depth = 37
        self.images = []
        self.masks = []
        for folder in image_folders:
            files = np.array(os.listdir(folder))
            images = np.sort(files[[('Mask' not in name and name != '.DS_Store') for name in files]])
            images = [folder + image for image in images]
            self.images.extend(images)
            
            masks = np.sort(files[['Mask' in name for name in files]])
            masks = [folder + mask for mask in masks]
            self.masks.extend(masks)

        if len(self.images) != len(self.masks):
            print('Different number of images and masks')

        self.transform = transform
    def __len__(self):
        return len(self.images) * self.image_depth
    
    # input must be a list
    # returns tensors that represent images/masks
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = nib.load(self.images[idx // self.image_depth]).get_fdata()
        image = image[:,:,idx % self.image_depth]
        
        mask = nib.load(self.masks[idx // self.image_depth]).get_fdata()
        mask = mask[:,:,idx % self.image_depth]
        sample = {'image': image, 'mask': mask}
        
        if self.transform:
            sample = self.transform(sample)

        return (sample['image'], sample['mask'])

# To use the full U-Net without losing data,
# we need the dimensions to be a multiple of 16
class CropTensor(object):
    def __init__(self, output_size):
        self.output_size = output_size
        
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        orig_shape = list(image.shape)
        start_ind = orig_shape[0] // 2 - self.output_size // 2
        end_ind = orig_shape[0] // 2 + self.output_size // 2
        
        # Channels for cross entropy loss
        image = image[np.newaxis, start_ind:end_ind, start_ind:end_ind]
        mask = mask[start_ind:end_ind, start_ind:end_ind]
        return {'image': image, 'mask': mask}


# modelled after the original UNet
class UNet(nn.Module):
    # This conv/relu combination results in no change in dimension for full image restoration
    def conv_relu(self, in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode=padding_mode
            ),
            nn.ReLU()
        )
    
    # This transpose doubles the dimensions
    def conv_transpose(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1):
        return nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        )
    
    def first_block(self, in_channels, out_channels):
        return nn.Sequential(
            self.conv_relu(in_channels, out_channels),
            self.conv_relu(out_channels, out_channels)
        )
    
    # Output: (x-4)/2
    def contract_block(self, in_channels, out_channels):
        # Testing: adding BatchNorm2d(out_channels) after ReLU layers
        return nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            self.conv_relu(in_channels, out_channels),
            self.conv_relu(out_channels, out_channels)
        )
    
    def bottleneck_block(self, in_channels, mid_channels, out_channels):
        return nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            self.conv_relu(in_channels, mid_channels),
            self.conv_relu(mid_channels, mid_channels),
            self.conv_transpose(mid_channels, out_channels)
        )
        
    # Output: (x-4)*2
    def expand_block(self, in_channels, mid_channels, out_channels):
        return nn.Sequential(
            self.conv_relu(in_channels, mid_channels),
            self.conv_relu(mid_channels, mid_channels),
            self.conv_transpose(mid_channels, out_channels)
        )

    def final_block(self, in_channels, mid_channels, out_channels):
        return nn.Sequential(
            self.conv_relu(in_channels, mid_channels),
            self.conv_relu(mid_channels, mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1)
        )
    
    def __init__(self):
        super().__init__()
        self.contraction = nn.ModuleList([
            # 288
            self.first_block(1, 64),
            # 288
            self.contract_block(64, 128),
            # 144
            self.contract_block(128, 256),
            # 72
            self.contract_block(256, 512),
            # 36
        ])
        
        self.bottleneck = self.bottleneck_block(512, 1024, 512)
        
        self.expansion = nn.ModuleList([
            # 36
            self.expand_block(1024, 512, 256),
            # 72
            self.expand_block(512, 256, 128),
            # 144
            self.expand_block(256, 128, 64),
            # 288
            self.final_block(128, 64, 2)
            # 288
        ])
        
        self.contraction_outputs = []

    def forward(self, image):
        for layer in self.contraction:
            image = layer(image)
            self.contraction_outputs.append(image)
        
        image = self.bottleneck(image)
        for i in range(4):
            image = torch.cat((self.contraction_outputs[3 - i], image), dim=1)
            image = self.expansion[i](image)
        self.contraction_outputs = []
        return image

def train_model2(model, optimizer, criterion, epochs):
    writer = SummaryWriter()
    VAL_INTERVAL = 5
    epoch_loss_values = []
    metric_values = []

    best_metric_epoch = -1

    phase = 'val'
    epoch = 0
    while epoch < NUM_EPOCHS:
        if epoch % VAL_INTERVAL == 0 and phase == 'train':
            epoch -= 1
            phase = 'val'
            print('-' * 10)
            print('Validation')

        else:
            phase = 'train'
            print('-' * 10)
            print('Epoch {}/{}'.format(epoch + 1, NUM_EPOCHS))
            lr = get_lr(optimizer)
            print('Learning Rate: ', lr)
            writer.add_scalar('learning_rate', lr, epoch + 1)

        if phase == 'train':
            model.train()
        else:
            model.eval()

        epoch_loss = 0
        step = 0
        for (image, mask) in data_loaders[phase]:
            step += 1
            inputs, labels = image.to(device, dtype=torch.float), mask.to(device, dtype=torch.long)
#             inputs = inputs.squeeze(4)
#             labels = labels.squeeze(4)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            epoch_loss += loss.item()
        epoch_loss /= step

        if phase == 'train':
            epoch_loss_values.append(epoch_loss)
            print('epoch {} average loss: {:.4f}'.format(epoch + 1, epoch_loss))

        else:
            metric_values.append(epoch_loss)
            best_metric = min(metric_values)
            if epoch_loss <= best_metric:
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), 'best_metric_model.pth')
                print('Saved new best metric model')
            print('val average loss: {:.4f}'.format(epoch_loss))
            print('best val loss: {:.4f} at epoch {}'.format(best_metric, best_metric_epoch))

        writer.add_scalar('loss/' + phase, epoch_loss, epoch + 1)
        epoch += 1
        
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

if __name__ == '__main__':
    print(os.listdir(datapath))
    all_animals = [12, 13, 18, 25, 27]
    train_set = LVDataset(all_animals[:4], transform=CropTensor(288))
    test_set = LVDataset(all_animals[5:], transform=CropTensor(288))
    train_set = LVDataset([12], transform=CropTensor(288))
    test_set = LVDataset([13], transform=CropTensor(288))
    print('Size of training set: ', len(train_set))
    print('Size of test set: ', len(test_set))
    print('Image Shape: ', train_set[0][0].shape)
    print('Mask Shape: ', train_set[0][1].shape)
    
    train_loader = DataLoader(
        train_set,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=TEST_BATCH_SIZE,
        shuffle=True,
    )

    data_loaders = {'train': train_loader, 'val': test_loader}
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    train_model2(model, optimizer, criterion, NUM_EPOCHS)