#!/usr/bin/env python
# coding: utf-8

# In[92]:


# importing the modules
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split

# importing the data
train = pd.read_csv('C:/Users/Big-Max/Desktop/BIG-MAX/BOOKS\Machine Learning/DATA SCIENCE/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('C:/Users/Big-Max/Desktop/BIG-MAX/BOOKS\Machine Learning/DATA SCIENCE/house-prices-advanced-regression-techniques/test.csv')
sub = pd.read_csv('C:/Users/Big-Max/Desktop/BIG-MAX/BOOKS\Machine Learning/DATA SCIENCE/house-prices-advanced-regression-techniques/sample_submission.csv')


# In[53]:


# filling the missing values in both the train and test with 0
x = train.fillna(0)
y = test.fillna(0)

# we display a summary of the data
train.describe()


# In[17]:


# we show a correlation between data
train.corr()


# In[297]:


df = x.loc[:,['OverallQual','OverallCond','YearBuilt','SalePrice']]
df.plot.area(x = 'OverallQual', y = 'SalePrice')


# In[44]:


# a scatter plot of relationship between OverallQual and SalePrice
df.plot.scatter(x = 'OverallQual', y = 'SalePrice')


# In[305]:


# relationship between overall condition and price
df.plot.scatter(x = 'OverallCond', y = 'SalePrice')


# In[74]:


x_2 = train.loc[:,['Id','MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF ','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvG','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold','SalePrice']]
x_3 = x_2.fillna(0)


# In[120]:


xx = x_3.drop(['SalePrice'], axis = 1)
yy = np.ravel(x_3['SalePrice'])


# In[190]:


# splitting the data
x_train,x_test,y_train,y_test = train_test_split(xx, yy, test_size= 0.4, random_state =42)


# In[191]:


############################### training the model with SUPPORT VECTOR MACHINE ####################################
from sklearn.svm import SVR

model = SVR(kernel = 'linear')

model.fit(x_train,y_train)


# In[192]:


# predicting the sales
pred = model.predict(x_test)
pred = pd.DataFrame(pred, columns=['Predicted_sales'])


# In[193]:


# measuring the model MSE
from sklearn.metrics import r2_score,mean_squared_error

mse = mean_squared_error(y_test, pred)
mse


# In[194]:


# RMSE
rmse = np.sqrt(mse)
rmse


# In[216]:


# SELECTING THE COLUMN ID
Id = x_3['Id']
Id = Id.dropna()
Id = pd.DataFrame(Id)


# In[220]:


# JOINING THE COLUMNS
result = pd.concat([Id, pred], axis = 1)


# In[221]:


# PREDICTING THE RESULTS
result = result.dropna()
result


# In[233]:


# TESTING WITH NEW DATA
new = xx.groupby('Id')
a = new.get_group(1460)


# In[237]:


# PREDICTING
z = model.predict(a)
z = pd.DataFrame(z)
z


# In[232]:


a


# In[279]:


##################################### new model GAUSSIAN ##############################################
from sklearn.naive_bayes import GaussianNB
model_2 = GaussianNB()


# In[289]:


# model training
model_2.fit(x_train, y_train)


# In[290]:


# model predicting
G_pred = model_2.predict(x_test)
G_pred = pd.DataFrame(G_pred, columns=['Predicted_sales'])


# In[291]:


from sklearn.metrics import r2_score,mean_squared_error
G_test = mean_squared_error(y_test,pred)


# In[292]:


G_test


# In[293]:


G_test_sq = np.sqrt(G_test)
G_test_sq


# In[294]:


resul_2 = pd.concat([Id, G_pred], axis = 1)
result_2 = resul_2.dropna()
result_2


# In[295]:


########################## LINEAR REGRESSION#####################################################
from sklearn.linear_model import LinearRegression
model_3 = LinearRegression(normalize=True)


# In[254]:


model_3.fit(x_train,y_train)


# In[272]:


LR_pred = model_3.predict(x_test)
LR_pred = pd.DataFrame(LR_pred, columns=['Predicted_sales'])


# In[273]:


from sklearn.metrics import r2_score,mean_squared_error
LR_mse = mean_squared_error(y_test,LR_pred)
LR_mse


# In[274]:


LR_rmse = np.sqrt(LR_mse)
LR_rmse


# In[275]:


resul_3 = pd.concat([Id,LR_pred], axis = 1)
result_3 =resul_3.dropna() 
result_3


# In[ ]:




