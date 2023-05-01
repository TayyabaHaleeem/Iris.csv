#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[14]:


df = pd.read_csv('Iris.csv')


# In[15]:


df.head()


# In[16]:


df.describe()


# In[17]:


df['Species'] = pd.factorize(df['Species'])[0]
df.plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', c='Species', colormap='viridis')


# In[18]:


# Visualize with matplotlib
fig, ax = plt.subplots()
colors = {'Iris-setosa':'r', 'Iris-versicolor':'g', 'Iris-virginica':'b'}
ax.scatter(df['SepalLengthCm'], df['SepalWidthCm'], c=df['Species'])


# In[19]:


# Visualize with seaborn
sns.scatterplot(data=df, x='SepalLengthCm', y='SepalWidthCm', hue='Species')


# In[20]:


sns.set(style="ticks")

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, orient="h", palette="Set2")
plt.title("Boxplot of Iris Dataset Features")
plt.show()


# In[24]:


sns.set(style="ticks")

sns.pairplot(df, hue="Species", height=2.5)
plt.show()


# In[10]:


# Visualize with plotly
import plotly.express as px
fig = px.scatter(df, x='SepalLengthCm', y='SepalWidthCm', color='Species')
fig.show()

