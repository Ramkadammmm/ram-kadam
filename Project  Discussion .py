#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[5]:


df = pd.read_csv('https://raw.githubusercontent.com/dsrscientist/DSData/master/Telecom_customer_churn.csv')
df


# This dsataset contains the details of customer in which both numerical and categorical data are present. Here 'churn' is the Target variable which contains two categorical so it will term as "classification problem" where we need to predict the several churn using the classification models
# 
# 

# In[6]:


df.head(15)


# In[7]:


# last 30 rows information
df.tail(30)


# In[8]:


df.columns


# In[9]:


df.columns.tolist()


# In[10]:


# checking the types of the columns #int64 == o value presnet
df.dtypes


# In[11]:


# checking the null values
df.isnull().sum()


# There is NO null value

# In[12]:


df.isnull().sum().sum()


# In[13]:


df.info()


# In[14]:


# lets visualize usin map
sns.heatmap(df.isnull())


# In[15]:


# To get good overview of the dataset
df.info()


# This gives the brief about the dataset which iclude indexing type, column type, no null values and memory usage
# 
# 

# In[16]:


df['TotalCharges'].unique()


# In[18]:


df['TotalCharges'].nunique()


# total No. of Unique value is 6531 out of 7043
# 
# 

# In[20]:


#checking the value counts of each column
for i in df.columns:
    print(df[i].value_counts())
    print("\n")


# These are the value counts of columns and we can see blank in TotalCharges column. Lets Check out the unique value of that column
# 
# 

# In[23]:


df['TotalCharges'].unique()


# In[26]:


df.shape[0]


# We can notice that "ToatalCharges" has continious data but its reflecting as object datatype and 11 record of this column has blank data Lets handle thid column
# 
# 

# In[27]:


# Checking the space in TotalCharges column
df.loc[df["TotalCharges"]==" "]


# By Locating the ToatlCharges we can find this coulmn has space as values but it was showing 0 missing values in this column. lets fill this column by some values.
# 
# 

# In[29]:


df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
df["TotalCharges"].value_counts()


# In[30]:


df.isnull().sum()


# Now there is 11 Null Values in "TotalCharges" because we replaced
# 
# 

# In[31]:


df.info()


# In[33]:


# converting object datatype to float datatype
df["TotalCharges"] = df["TotalCharges"].astype(float)
df["TotalCharges"].dtype


# In[34]:


df.info()



# We have converted the datatype of TotalCharges from object to float
# 
# 

# In[35]:


# check the null values
df.isnull().sum()


# In[36]:


# replacing NAN values using mean method
np.mean(df['TotalCharges'])


# In[37]:


df.iloc[488:500,:]


# In[38]:


# checking the mean of TotalCharges column
print("The Mean Value of TotalCharges is:", df["TotalCharges"].mean())


# In[39]:


df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())


# In[40]:


df.isnull().sum()


# In[45]:


sns.heatmap(df.isnull(), cmap = "cool_r")


# In[47]:


df.iloc[488,:]


# Now we can see there are no missing values in any of the column

# In[48]:


# separating the Numerical and categorical columns
#Checking for categorical columns
categorical_col = []
for i in df.dtypes.index:
    if df.dtypes[i] == "object":
        categorical_col.append(i)
print("Categorical Columns: ", categorical_col)
print("\n")

# checking for Numerical columns
numerical_col = []
for i in df.dtypes.index:
    if df.dtypes[i]!="object":
        numerical_col.append(i)
print("Numerical Columns: ", numerical_col)


# In[49]:


# checking number of unique values in each column
df.nunique().to_frame("No. of unique values")


# In[ ]:




