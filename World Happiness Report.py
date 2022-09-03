#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('https://raw.githubusercontent.com/dsrscientist/DSData/master/happiness_score_dataset.csv')
df


# In[3]:


type(df.columns)


# In[4]:


df.columns


# In[5]:


df.shape


# # Null values

# In[6]:


df.isnull().sum()


# # Dataframe description:

# In[8]:


df.dtypes


# In[17]:


df.loc[df['HappinessScore']==" "]


# In[ ]:


df.isnull().sum()


# # Making DataFrame for the Nominal Data

# In[11]:


df_visualization_nominal=df[['Country', 'Region', 'Happiness Rank', 'Happiness Score',
       'Standard Error', 'Economy (GDP per Capita)', 'Family',
       'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)',
       'Generosity', 'Dystopia Residual']].copy()


# In[12]:


df_visualization_nominal.columns


# # EDA

# In[20]:


df.columns


# In[21]:


import seaborn as sns
import matplotlib.pyplot as plt
import warnings                      # to ignore any warnings
warnings.filterwarnings("ignore")


# In[26]:


ax = sns.countplot(x="Happiness Score", data=df_visualization_nominal)
print(df_visualization_nominal["Happiness Score"].value_counts())


# In[27]:


ax = sns.countplot(x="Country", data=df_visualization_nominal)
print(df_visualization_nominal["Country"].value_counts())


# In[28]:


ax = sns.countplot(x="Region", data=df_visualization_nominal)
print(df_visualization_nominal["Region"].value_counts())


# In[29]:


ax = sns.countplot(x="Happiness Rank", data=df_visualization_nominal)
print(df_visualization_nominal["Happiness Rank"].value_counts())


# In[30]:


ax = sns.countplot(x="Standard Error", data=df_visualization_nominal)
print(df_visualization_nominal["Standard Error"].value_counts())


# In[31]:


ax = sns.countplot(x="Economy (GDP per Capita)", data=df_visualization_nominal)
print(df_visualization_nominal["Economy (GDP per Capita)"].value_counts())


# In[32]:


ax = sns.countplot(x="Family", data=df_visualization_nominal)
print(df_visualization_nominal["Family"].value_counts())


# In[33]:


ax = sns.countplot(x="Health (Life Expectancy)", data=df_visualization_nominal)
print(df_visualization_nominal["Health (Life Expectancy)"].value_counts())


# In[34]:


ax = sns.countplot(x="Freedom", data=df_visualization_nominal)
print(df_visualization_nominal["Freedom"].value_counts())


# In[36]:


ax = sns.countplot(x="Trust (Government Corruption)", data=df_visualization_nominal)
print(df_visualization_nominal["Trust (Government Corruption)"].value_counts())


# In[37]:


ax = sns.countplot(x="Generosity", data=df_visualization_nominal)
print(df_visualization_nominal["Generosity"].value_counts())


# In[38]:


ax = sns.countplot(x="Dystopia Residual", data=df_visualization_nominal)
print(df_visualization_nominal["Dystopia Residual"].value_counts())


# In[39]:


from sklearn.preprocessing import OrdinalEncoder
enc=OrdinalEncoder()


# In[40]:


for i in df.columns:
    if df[i].dtypes=="object":
        df[i]=enc.fit_transform(df[i].values.reshape(-1,1))


# In[41]:


df


# In[42]:


#Describe the dataset:


# In[43]:


df.describe()


# In[45]:


import matplotlib.pyplot as plt
plt.figure(figsize=(18,12))
sns.heatmap(df.describe(),annot=True,linewidth =0.5,linecolor='black',fmt='.2f')


# In[46]:


df.corr()['Happiness Score'].sort_values()


# In[47]:


import matplotlib.pyplot as plt
plt.figure(figsize=(18,12))

sns.heatmap(df.corr(), annot=True,linewidth =0.5,linecolor='black',fmt='.2f')


# In[49]:


df.columns


# In[51]:


df['Country'].plot.box()


# In[52]:


df['Region'].plot.box()


# In[53]:


df['Happiness Rank'].plot.box()


# In[54]:


df['Standard Error'].plot.box()


# In[ ]:




