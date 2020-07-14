#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.stats as stat

from tkinter import *

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics  
from sklearn.metrics import r2_score


# # A)	Data Manipulation:
# a.	Extract the 5th column & store it in ‘customer_5’
# b.	Extract the 15th column & store it in ‘customer_15’
# c.	Extract all the male senior citizens whose Payment Method is Electronic check & store the result in ‘senior_male_electronic’
# d.	Extract all those customers whose tenure is greater than 70 months or their Monthly charges is more than 100$ & store the result in ‘customer_total_tenure’
# e.	Extract all the customers whose Contract is of two years, payment method is Mailed check & the value of Churn is ‘Yes’ & store the result in ‘two_mail_yes’
# f.	Extract 333 random records from the customer_churn dataframe & store the result in ‘customer_333’
# g.	Get the count of different levels from the ‘Churn’ column
# 

# In[2]:


customer_churn= pd.read_csv('customer_churn.csv')
customer_churn.head()


# In[4]:


customer_5= customer_churn.iloc[:,4]
customer_5.head()
customer_15=customer_churn.iloc[:,14]
customer_15.head()


# In[5]:


senior_male_electronic=customer_churn[(customer_churn['gender']=='Male') & (customer_churn['SeniorCitizen']==1) & (customer_churn['PaymentMethod']=='Electronic check')]
senior_male_electronic.head()


# In[6]:


customer_total_tenure=customer_churn[(customer_churn['tenure']>70) | (customer_churn['MonthlyCharges']>100)]
customer_total_tenure.head()


# In[7]:


two_mail_yes=customer_churn[(customer_churn['Contract']=='Two year') & (customer_churn['PaymentMethod']=='Mailed check')&(customer_churn['Churn']=='Yes') ]
two_mail_yes


# In[8]:


customer_333=customer_churn.sample(n=333)
customer_333.head()


# In[9]:


customer_churn['Churn'].value_counts()


# # B)	Data Visualization:

# a.	Build a bar-plot for the ’InternetService’ column:
# i.	Set x-axis label to ‘Categories of Internet Service’
# ii.	Set y-axis label to ‘Count of Categories’
# iii.	Set the title of plot to be ‘Distribution of Internet Service’
# iv.	Set the color of the bars to be ‘orange’
# 

# In[10]:


y=customer_churn['InternetService'].value_counts()
x=customer_churn['InternetService'].unique()
plt.bar(x,y,color="Orange")
plt.title('Distribution of Internet Service')
plt.show()
#customer_churn['InternetService'].values
#x.count()


# b.	Build a histogram for the ‘tenure’ column:
# i.	Set the number of bins to be 30
# ii.	Set the color of the bins  to be ‘green’
# iii.	Assign the title ‘Distribution of tenure’
# 

# In[12]:


x=customer_churn['tenure']
plt.hist(x,bins=30,color='green')
#plt.title('Distribution of tenure',y=-0.2) # sets the title at bottom
plt.title('Distribution of tenure')
plt.show()


# c.	Build a scatter-plot between ‘MonthlyCharges’ & ‘tenure’. Map ‘MonthlyCharges’ to the y-axis & ‘tenure’ to the ‘x-axis’:
# i.	Assign the points a color of ‘brown’
# ii.	Set the x-axis label to ‘Tenure of customer’
# iii.	Set the y-axis label to ‘Monthly Charges of customer’
# iv.	Set the title to ‘Tenure vs Monthly Charges’
# 

# In[13]:


x=customer_churn['tenure']
x=x.sample(n=300)
y=customer_churn['MonthlyCharges']
y=y.sample(n=300)
plt.scatter(x,y,color='brown')
plt.xlabel('Tenure of customer')
plt.ylabel('Monthly Charges of customer')
plt.title('Tenure vs Monthly Charges')
plt.show()


# d.	Build a box-plot between ‘tenure’ & ‘Contract’. Map ‘tenure’ on the y-axis & ‘Contract’ on the x-axis. 

# In[5]:


x=customer_churn['Contract']
#x=x.sample(n=300)
y=customer_churn['tenure']
#y=y.sample(n=300)
#plt.boxplot(y)
#plt.show()
customer_churn.boxplot(column=['tenure'],by=['Contract'],grid=False)


# # C)	Linear Regression:

# a.	Build a simple linear model where dependent variable is ‘MonthlyCharges’ and independent variable is ‘tenure’
# i.	Divide the dataset into train and test sets in 70:30 ratio. 
# ii.	Build the model on train set and predict the values on test set
# iii.	After predicting the values, find the root mean square error
# iv.	Find out the error in prediction & store the result in ‘error’
# v.	Find the root mean square error

# In[9]:


x=pd.DataFrame(customer_churn['tenure'])
y=pd.DataFrame(customer_churn['MonthlyCharges'])

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3,random_state=0)
regressor= LinearRegression()
print(x_train.shape)
regressor.fit(x_train,y_train)


# In[16]:


y_pred=regressor.predict(x_test)

print(pd.DataFrame(y_pred).head(5))
print(pd.DataFrame(y_test).head(5))
#print(pd.DataFrame(y_pred).head())
#x_test


# In[13]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
r2_score(y_test,y_pred,sample_weight=None)


# In[18]:


from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_pred, y_test)
rmse = np.sqrt(mse)
rmse


# In[51]:


print(pd.DataFrame(y_pred).head(5))
print(pd.DataFrame(y_test).head(5))


# # D)	Logistic Regression:

# a.	Build a simple logistic regression model where dependent variable is ‘Churn’ & independent variable is ‘MonthlyCharges’
# i.	Divide the dataset in 65:35 ratio
# ii.	Build the model on train set and predict the values on test set
# iii.	Build the confusion matrix and get the accuracy score
# 

# In[30]:


x=pd.DataFrame(customer_churn['MonthlyCharges'])
y=pd.DataFrame(customer_churn['Churn'])
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.35,random_state=1)
logReg= LogisticRegression()
logReg.fit(x_train,y_train)


# In[31]:


y_pred = logReg.predict(x_test)
#logReg.score(x_test,y_pred)


# In[32]:


from sklearn.metrics import confusion_matrix, accuracy_score
confu_matrix = confusion_matrix(y_pred, y_test)
accuracy= accuracy_score(y_pred,y_test)
print(confu_matrix,accuracy)


# #--------------Multiple logistic regression-------------------

# b.	Build a multiple logistic regression model where dependent variable is ‘Churn’ & independent variables are ‘tenure’ & ‘MonthlyCharges’
# i.	Divide the dataset in 80:20 ratio
# ii.	Build the model on train set and predict the values on test set
# iii.	Build the confusion matrix and get the accuracy score
# 

# In[57]:


y= customer_churn['Churn']
x=pd.DataFrame(customer_churn.loc[:,['MonthlyCharges','tenure']])
#x= pd.DataFrame(data=customer_churn,columns=['tenure','MonthlyCharges'])


# In[61]:


x_train,x_test,y_train,y_test= train_test_split(x,y,train_size=0.8)
multiLogModel= LogisticRegression()
multiLogModel.fit(x_train,y_train)


# In[62]:


y_pred=multiLogModel.predict(x_test)


# In[63]:


from sklearn.metrics import confusion_matrix, accuracy_score
confu_matrix = confusion_matrix(y_pred, y_test)
accuracy= accuracy_score(y_pred,y_test)
print(confu_matrix,accuracy)


# # E)	Decision Tree:

# a.	Build a decision tree model where dependent variable is ‘Churn’ & independent variable is ‘tenure’
# i.	Divide the dataset in 80:20 ratio
# ii.	Build the model on train set and predict the values on test set
# iii.	Build the confusion matrix and calculate the accuracy
# 

# In[10]:


x= pd.DataFrame(customer_churn['tenure'])
y= customer_churn['Churn']

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)


# In[11]:


from sklearn.tree import DecisionTreeClassifier
decisionTree= DecisionTreeClassifier()
decisionTree.fit(x_train,y_train)


# In[12]:


y_pred= decisionTree.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score
confu_matrix = confusion_matrix(y_test, y_pred)
accuracy= accuracy_score(y_test,y_pred)
print(confu_matrix,accuracy)


# # F)	Random Forest:

# a.	Build a Random Forest model where dependent variable is ‘Churn’ & independent variables are ‘tenure’ and ‘MonthlyCharges’
# i.	Divide the dataset in 70:30 ratio
# ii.	Build the model on train set and predict the values on test set
# iii.	Build the confusion matrix and calculate the accuracy
# 

# In[32]:


#x=pd.DataFrame(customer_churn.loc[:,['tenure','MonthlyCharges']])
x=customer_churn[['tenure','MonthlyCharges']]
y= customer_churn['Churn']
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)


# In[33]:


from sklearn.ensemble import RandomForestClassifier
rndm= RandomForestClassifier(n_estimators=100)

rndm.fit(x_train,y_train)


# In[34]:


y_pred= rndm.predict(x_test)


# In[35]:


from sklearn.metrics import confusion_matrix,accuracy_score

conf= confusion_matrix(y_test,y_pred)
acc= accuracy_score(y_test,y_pred)
print(conf,acc)


# In[ ]:




