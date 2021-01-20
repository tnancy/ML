#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r"Absenteeism.csv")
features = ['Transportation expense', 'Distance from Residence to Work', 'Weight', 'Height']
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['Day of the week']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['PCA1', 'PCA2'])

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('PCA1', fontsize = 15)
ax.set_ylabel('PCA2', fontsize = 15)
ax.set_title('Absenteesim', fontsize = 20)
targets = ['3', '5', '2']
colors = ['r', 'g', 'b']
finalDf = pd.concat([principalDf,df[['Day of the week']]],axis=1)
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Day of the week'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'PCA1']
               , finalDf.loc[indicesToKeep, 'PCA2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
pca.explained_variance_ratio_


# In[ ]:




