#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('https://raw.githubusercontent.com/dsrscientist/dataset3/main/Salaries.csv')
df


# In[3]:


type(df.columns)


# In[4]:


df.columns


# In[29]:


df.head()


# # Null values

# In[5]:


df.shape


# In[6]:


df.isnull().sum()


# In[31]:


df.columns


# In[32]:


#there are no null values.


# # Dataframe description:

# In[ ]:


#checking the datatypes of the columns


# In[34]:


df.dtypes


# In[35]:


df_visualization_nominal=df[['rank', 'discipline', 'yrs.since.phd', 'yrs.service', 'sex', 'salary']].copy()


# In[36]:


df_visualization_nominal.columns


# # EDA
# 
# Scatterplot:

# In[8]:


df.columns


# In[37]:


import seaborn as sns


# In[38]:


ax = sns.countplot(x="rank", data=df_visualization_nominal)
print(df_visualization_nominal["rank"].value_counts())


# In[39]:


ax = sns.countplot(x="discipline", data=df_visualization_nominal)
print(df_visualization_nominal["discipline"].value_counts())


# In[40]:


ax = sns.countplot(x="yrs.since.phd", data=df_visualization_nominal)
print(df_visualization_nominal["yrs.since.phd"].value_counts())


# In[41]:


ax = sns.countplot(x="yrs.service", data=df_visualization_nominal)
print(df_visualization_nominal["yrs.service"].value_counts())


# In[42]:


ax = sns.countplot(x="sex", data=df_visualization_nominal)
print(df_visualization_nominal["sex"].value_counts())


# In[43]:


from sklearn.preprocessing import OrdinalEncoder
enc=OrdinalEncoder()


# In[45]:


for i in df.columns:
    if df[i].dtypes=="object":
        df[i]=enc.fit_transform(df[i].values.reshape(-1,1))


# In[46]:


df


# In[ ]:


Describe the dataset:


# In[48]:


#only continous columns.
df.describe()


# In[49]:


import matplotlib.pyplot as plt
plt.figure(figsize=(7,3))
sns.heatmap(df.describe(),annot=True,linewidth =0.5,linecolor='black',fmt='.2f')


# In[50]:


df.corr()['salary'].sort_values()


# In[51]:


import matplotlib.pyplot as plt
plt.figure(figsize=(7,3))

sns.heatmap(df.corr(), annot=True,linewidth =0.5,linecolor='black',fmt='.2f')


# In[52]:


df.skew()


# In[56]:


#Outliers Check


# In[57]:


df['rank'].plot.box()


# In[58]:


df['discipline'].plot.box()


# In[59]:


df['yrs.since.phd'].plot.box()


# In[60]:


df['yrs.service'].plot.box()


# In[61]:


df['sex'].plot.box()


# In[62]:


df['salary'].plot.box()


# In[63]:


#Considering the outlier removal


# In[64]:


df.shape


# In[65]:


from scipy.stats import zscore
import numpy as np
z=np.abs(zscore(df))
threshold=3
np.where(z>3)


# In[66]:


df_new_z=df[(z<3).all(axis=1)]
df_new_z


# In[67]:


df_new_z.shape


# In[ ]:


##percentage loss of data


# In[68]:


data_loss=(397-354)/397*100


# In[69]:


data_loss


# In[ ]:


#Separating columns into features and target:


# In[70]:


features=df.drop("salary",axis=1)
target =df["salary"]


# In[ ]:


# Scaling the data using Min-Max scaler:


# In[71]:


from sklearn.preprocessing import MinMaxScaler
mms=MinMaxScaler()
from sklearn.linear_model import LinearRegression
lr = LinearRegression
from sklearn.metrics  import r2_score
from sklearn.model_selection import train_test_split


# In[80]:


import warnings
warnings.filterwarnings('ignore')


# In[81]:


features.head()
target.head()

for i in range(0,100):
    features_train,features_test,target_train,target_test=train_test_split(features,target,test_size=0.2,random_state=i)
    
    lr.fit(features,target)
    pred_train = lr.predict(features_train)
    pred_test = lr.predict(features_test)
    print(f"At random state{i},the training accuracy is:-{r2_score(target_train,pred_train)}")
    print(f"At random state{i},the testing accuracy is:-{r2_score(target_test,pred_test)}")
    print("\n")
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




