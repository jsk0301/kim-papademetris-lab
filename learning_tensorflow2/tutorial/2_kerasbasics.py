#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


print(tf.__version__)


# In[3]:


# Database of 60000 training images of clothes and 10000 test images
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# In[4]:


# Names for test labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# In[5]:


train_images.shape


# In[6]:


len(train_labels)


# In[7]:


test_images.shape


# In[8]:


len(test_labels)


# In[9]:


# Showing the first image
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()


# In[10]:


# Since each pixel value ranges from 0 to 255, scale to 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0


# In[11]:


# Viewing first 25 images to confirm scaling
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# In[ ]:


# Neural networks are made up of layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Transforms 28 x 28 array to 1D
    keras.layers.Dense(128, activation='relu'),  # Fully connected layer with 128 nodes
    keras.layers.Dense(10, activation='softmax') # Fully connected 10-node layer
])


# In[ ]:




