#!/usr/bin/env python
# coding: utf-8

# In[28]:


#importing the necessary libraries.

import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
from skimage import color
from skimage import io

image_raw = imread("flower.jpg")
print(image_raw.shape)

plt.figure(figsize=[12,8])
plt.imshow(image_raw)


# In[29]:



img = color.rgb2gray(io.imread('flower.jpg'))
plt.figure(figsize=[12,8])
plt.imshow(image_bw)


# In[31]:



pca = PCA()
pca.fit(img)

variance = np.cumsum(pca.explained_variance_ratio_)*100

# Calculating the number of components needed to preserve 98% of the data
k = np.argmax(variance>98)
print("k for 98% variance: "+ str(k))
#print("\n")

plt.figure(figsize=[10,5])
plt.axvline(x=k, color="k")
plt.axhline(y=95, color="r")
ax = plt.plot(var_cumu)


# In[35]:


ipca = IncrementalPCA(n_components=k)
image_compressed = ipca.inverse_transform(ipca.fit_transform(img))

# Plotting the compressed image
plt.figure(figsize=[12,8])
plt.imshow(image_compressed)


# In[ ]:




