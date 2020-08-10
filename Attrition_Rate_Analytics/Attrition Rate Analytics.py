#!/usr/bin/env python
# coding: utf-8

# # Attrition Rate Analytics

# In[1]:


#Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


#Import dataset
df = pd.read_csv(r"C:\Users\KRISH\Downloads\TelcoCustomer.csv")
df.head(10).T


# In[3]:


df.info()


# **Target Variable 'Churn' signifies Attrition Rate**

# In[4]:


#Get the number of customers that churned
df['Churn'].value_counts()


# In[5]:


#Visualize the count of customer churn
sns.countplot(df['Churn'])


# In[6]:


#Percentage of customers leaving 
retained = df[df.Churn == 'No']
churned = df[df.Churn == 'Yes']
num_retained = retained.shape[0]
num_churned = churned.shape[0]
#Percentage of customers that Stayed with the company
print( num_retained / (num_retained + num_churned) * 100 , "% of customers Stayed with the company.")
#Percentage of customers that left the company
print( num_churned / (num_retained + num_churned) * 100,"% of customers Left the company.")


# In[7]:


#Converting the total charges from object type to numeric
df['TotalCharges'] = df['TotalCharges'].replace(r'\s+',np.nan,regex=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])


# In[8]:


df.Partner.value_counts(normalize=True).plot(kind='pie',startangle=90, autopct='%1.1f%%')


# In[9]:


colors_list = ['#d9534f','#5bc0de']
df.SeniorCitizen.value_counts(normalize=True).plot(kind='bar',color = colors_list,edgecolor=None)


# In[10]:


colors_list = ['#483D8B','#FF1493']
df.gender.value_counts(normalize=True).plot(kind='bar',color = colors_list,edgecolor=None)


# In[11]:


df.tenure.value_counts(normalize=True).plot(kind='bar',figsize=(16,7))


# In[12]:


colors_list = ['#00C957','#FF3030']
df.PhoneService.value_counts(normalize=True).plot(kind='bar',color = colors_list,edgecolor=None)


# In[13]:


df.MultipleLines.value_counts(normalize=True).plot(kind='pie',startangle=90, autopct='%1.1f%%')


# In[14]:


df.InternetService.value_counts(normalize=True).plot(kind='pie',startangle=180, autopct='%1.1f%%')


# In[15]:


df.Contract.value_counts(normalize=True).plot(kind='pie',startangle=180, autopct='%1.1f%%')


# In[16]:


df.PaymentMethod.value_counts(normalize=True).plot(kind='pie',startangle=180, autopct='%1.1f%%')


# **Data Analysis: Analyzing other variables with repect to the Target variable**

# In[17]:


#Target Variable 'Churn'=Attrition Rate
colors_list = ['#d9534f','#5bc0de']
df.Churn.value_counts(normalize=True).plot(kind='bar',color = colors_list,edgecolor=None)


# In[18]:


#Gender Vs Attrition Rate
print(pd.crosstab(df.gender,df.Churn,margins=True))
pd.crosstab(df.gender,df.Churn,margins=True).plot(kind='bar')


# In[19]:


print('Percent of Females that left the Company {0}'.format((939/1869)*100))
print('Percent of Males that left the Company {0}'.format((930/1869)*100))


# **Gender does'nt play an important role in predicting our Target variable.**

# In[20]:


#Contract Vs Attrition Rate
print(pd.crosstab(df.Contract,df.Churn,margins=True))
pd.crosstab(df.Contract,df.Churn,margins=True).plot(kind='bar')


# In[21]:


print('Percent of Month-to-Month Contract People that left the Company {0}'.format((1655/1869)*100))
print('Percent of One Year Contract People that left the Company {0}'.format((166/1869)*100))
print('Percent of Two Year Contract People that left the Company {0}'.format((48/1869)*100))


# **Most of the People that left were the ones who had Month-to-Month Contract.**

# In[22]:


#InternetService Vs Attrition Rate
print(pd.crosstab(df.InternetService,df.Churn,margins=True))
pd.crosstab(df.InternetService,df.Churn,margins=True).plot(kind='bar')


# In[23]:


print('Percent of DSL Internet-Service People that left the Company {0}'.format((459/1869)*100))
print('Percent of Fiber Optic-Internet Service People that left the Company {0}'.format((1297/1869)*100))
print('Percent of No Internet-Service People that left the Company {0}'.format((113/1869)*100))


# **Most of the People that left had Fiber-Optic Internet-Service.**

# In[24]:


#Tenure Median Vs Attrion Rate
print(pd.crosstab(df.tenure.median(),df.Churn))
pd.crosstab(df.tenure.median(),df.Churn).plot(kind='bar',figsize=(7,5));


# In[25]:


#Partner Vs Attrition Rate
print(pd.crosstab(df.Partner,df.Dependents,margins=True))
pd.crosstab(df.Partner,df.Dependents,margins=True).plot(kind='bar',figsize=(5,5))


# In[26]:


print('Percent of Partners that had Dependents {0}'.format((1749/2110)*100))
print('Percent of Non-Partner that had Dependents {0}'.format((361/2110)*100))


# **Partners had a much larger percent of Dependents than Non-Partner tells us that Most Partners might be married.**

# In[27]:


#Partner Vs Attrition Rate
print(pd.crosstab(df.Partner,df.Churn,margins=True))
pd.crosstab(df.Partner,df.Churn,margins=True).plot(kind='bar',figsize=(5,5))


# In[28]:


plt.figure(figsize=(17,8))
sns.countplot(x=df['tenure'],hue=df.Partner)


# **Most of the People that were Partner will stay Longer with the Company.**

# In[29]:


#SeniorCitizen Vs Attrition Rate
print(pd.crosstab(df.SeniorCitizen,df.Churn,margins=True))
pd.crosstab(df.SeniorCitizen,df.Churn,normalize=True).plot(kind='bar')


# **Checking for Outliers in Monthly Charges and Total Charges using Box Plots**

# In[30]:


df.boxplot('MonthlyCharges')


# In[31]:


df.boxplot('TotalCharges')


# **Both Monthly Charges and Total Charges don't have any Outliers.**

# In[32]:


df.describe()


# In[33]:


#Correlation Matrix
sns.heatmap(df.corr(),xticklabels=df.corr().columns.values,yticklabels=df.corr().columns.values,annot=True)


# **We can see that Tenure and Total Charges are correlated and also Monthly Charges and Total Charges are also correlated with each other.So this proves our first Hypothesis correct i.e Total Charges = Monthly Charges x Tenure + Additional Tax that we had taken above.** 

# **Data Munging**

# In[34]:


#NULL Checking
df.isnull().sum()


# In[35]:


#Filling 11 NULL values in TotalCharges
fill = df.MonthlyCharges*df.tenure
df.TotalCharges.fillna(fill,inplace=True)
df.isnull().sum()


# **When Attrition Rate = Yes, Median Charges in $** 

# In[36]:


df.loc[(df.Churn == 'Yes'),'MonthlyCharges'].median()


# In[37]:


df.loc[(df.Churn == 'Yes'),'TotalCharges'].median()


# In[38]:


df.loc[(df.Churn == 'Yes'),'tenure'].median()


# In[39]:


df.loc[(df.Churn == 'Yes'),'PaymentMethod'].value_counts(normalize=True)


# In[40]:


df.loc[(df.Churn == 'Yes'),'PaperlessBilling'].value_counts(normalize=True)


# In[41]:


df.loc[(df.Churn == 'Yes'),'DeviceProtection'].value_counts(normalize=True)


# In[42]:


df.loc[(df.Churn == 'Yes'),'OnlineBackup'].value_counts(normalize=True)


# In[43]:


df.loc[(df.Churn == 'Yes'),'TechSupport'].value_counts(normalize=True)


# In[44]:


df.loc[(df.Churn == 'Yes'),'OnlineSecurity'].value_counts(normalize=True)


# **Encoding the Categorical Variables with Numeric using the get dummies Property which will make it easy for the Machine to make correct Prediction.**

# In[45]:


df = pd.get_dummies(df, columns = ['Contract','Dependents','DeviceProtection','gender',
                                                        'InternetService','MultipleLines','OnlineBackup',
                                                        'OnlineSecurity','PaperlessBilling','Partner',
                                                        'PaymentMethod','PhoneService','SeniorCitizen',
                                                        'StreamingMovies','StreamingTV','TechSupport'],
                              drop_first=True) 


# In[46]:


#Convert our Target Variable 'Churn' for Yes or No to 1 or 0
df = pd.get_dummies(df, columns = ['Churn'], drop_first=True)


# In[47]:


#Perform Feature Scaling and One Hot Encoding
from sklearn.preprocessing import StandardScaler

#Perform Feature Scaling on 'tenure', 'MonthlyCharges', 'TotalCharges' in order to bring them on same scale
standardScaler = StandardScaler()
columns_for_ft_scaling = ['tenure', 'MonthlyCharges', 'TotalCharges']

#Apply the feature scaling operation on dataset using fit_transform() method
df[columns_for_ft_scaling] = standardScaler.fit_transform(df[columns_for_ft_scaling])


# In[48]:


df.head()


# In[49]:


df.columns


# **Number of columns increased and have suffixes attached, as a result of get_dummies method.**

# **Data Modelling**

# In[50]:


#Create Feature variable X and Target variable y
y = df['Churn_Yes']
X = df.drop(['Churn_Yes','customerID'], axis = 1)


# In[51]:


#Split the data into training set (80%) and test set (20%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# **Using Logistic Regression Model as our Target Variable has Binary Outcome**

# In[52]:


#Machine Learning classification model library
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[53]:


#Fit the logistic Regression model
logmodel = LogisticRegression(random_state=50)
logmodel.fit(X_train,y_train)

#Predict the value for new, unseen data
pred = logmodel.predict(X_test)

#Find Accuracy using accuracy_score method
logmodel_accuracy = round(metrics.accuracy_score(y_test, pred) * 100, 2)


# In[54]:


print(f'Accuracy for Logistic Regression model is {logmodel_accuracy}%')


# **Generate Confusion Matrix**

# In[55]:


from sklearn.metrics import confusion_matrix
conf_mat_logmodel = confusion_matrix(y_test,pred)
conf_mat_logmodel


# **Check Precision, recall, f1-score**

# In[56]:


from sklearn.metrics import classification_report
print( classification_report(y_test, pred) )


# **From the report, We can see that the recall of the model is about 90% meaning the model correctly identified about 90% of the customers that were retained and missed about 10%.
# The Precision of the model is about 86% and the f1-score is about 88%. The Accuracy of the model is about 82.11% which is better than the 73.46% that We could’ve done just by guessing a customer would always stay with the company.**

# In[57]:


#Predict the probability of Attrition of each customer
df['Probability_Attrition'] = logmodel.predict_proba(df[X_test.columns])[:,1]


# **Final Dataframe showcasing Probability of Attrition of each customer**

# In[58]:


df[['customerID','Probability_Attrition']]


# **Conclusion: The importance of this type of analysis is to help Companies make more Profit. Hence, this analysis aimed to build a model that Predicts the Attrition Rate of customers in a Telecom Company.**
