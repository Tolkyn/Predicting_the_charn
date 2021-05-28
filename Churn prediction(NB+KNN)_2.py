#!/usr/bin/env python
# coding: utf-8

# In[98]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings("ignore")


# In[99]:


data = pd.read_csv("C:\\Users\\tshynarbekova\\OneDrive - Schlumberger\\Desktop\\ML_by_Park\\final project\\telecom_users.csv")
data .head()


# In[100]:


data.columns


# In[101]:



data.shape


# In[102]:


data.info()


# In[103]:


data.describe()


# In[104]:


data=data.drop("Unnamed: 0", axis=1)


# In[105]:


data.head()


# In[106]:


data["customerID"]=data.customerID.str.split('-').str[0]


# In[107]:


data.head()


# In[108]:


data_cols=["gender","Partner","Dependents","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod"]
num_cols= ['SeniorCitizen', 'tenure', 'MonthlyCharges']


# In[109]:


plt.figure(figsize=(18, 18))
n=1
for col in ['tenure', 'MonthlyCharges']:
    plt.subplot(4,2,n)
    sns.boxplot(x='Churn', y= col, data=data)
    plt.title('Variation of  ' + col)
    n=n+1


# In[110]:


# Converting customer ID from object data type into numeric data 

def convert_x(x): 
    try: 
        return float(x) 
    except: 
        return np.NAN
data["TotalCharges"]=data["TotalCharges"].apply(convert_x)
data["customerID"]=data["customerID"].apply(convert_x)


# In[111]:


x=data["TotalCharges"].median()
data["TotalCharges"]= data["TotalCharges"].fillna(x)


# In[112]:


data.isnull().sum()


# In[113]:


from sklearn.preprocessing import LabelEncoder 
le=LabelEncoder()


# In[114]:


for i in data_cols: 
    data[i]=le.fit_transform(data[i])


# In[115]:


data


# In[116]:


data["Churn"]=le.fit_transform(data["Churn"])


# In[117]:


data.head()


# In[118]:


data.info()


# >> Above  we can see that our all features have numeric type(int or float ), here no feature with object type. Cool, so then we can to move splitting and prepere data to ML algorithms 

# In[119]:


#splitting data sets 
X=data.iloc[:, :-1].values
y=data.iloc[:, -1].values


# In[120]:


# splitting data set into training set and test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.25, random_state=0 )


#  # Naive Bayes algorithm

# In[121]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
classifier_2 = GaussianNB()
classifier_2.fit(X_train, y_train)


# In[122]:


y_pred_2 = classifier_2.predict(X_test)
print(np.concatenate((y_pred_2.reshape(len(y_pred_2),1), y_test.reshape(len(y_test),1)),1)) 
#? 


# In[123]:


conf_matrix = confusion_matrix(y_test, y_pred_2)
print("Confusion matrix is: \n", conf_matrix)

print("Accuracy is: ", round(accuracy_score(y_test, y_pred_2), 2))


# # KNN algorithm 

# In[124]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
classifier_3 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier_3.fit(X_train, y_train)


# In[125]:


y_pred_3 = classifier_3.predict(X_test)
print(np.concatenate((y_pred_3.reshape(len(y_pred_3),1), y_test.reshape(len(y_test),1)),1))


# In[126]:


conf_matrix_ = confusion_matrix(y_test, y_pred_3)
print("Confusion matrix: \n", conf_matrix_)
print("Accuracy is: \n", round(accuracy_score(y_test, y_pred_3), 2) )


# In[ ]:





# In[ ]:




