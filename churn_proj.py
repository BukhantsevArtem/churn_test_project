import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Data import
df = pd.read_csv('Telco-Customer-Churn.csv')
df.isnull().sum()
df.head()
df.drop('customerID', axis = 1, inplace=True )
df.info()
df.describe()

plt.figure(figsize=(12,6), dpi =200)
sns.heatmap(data = df.corr(), annot = True )

sns.countplot(data = df, x = 'Churn')

plt.figure(figsize = (12,6), dpi = 200)
sns.boxplot(data = df, x = 'Contract', y = 'TotalCharges', hue = 'Churn')
#plt.legend(loc=(1.1,0.5))



corr_df  = pd.get_dummies(df[['gender', 'SeniorCitizen', 'Partner', 'Dependents','PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport','StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod','Churn']]).corr()

corr_df['Churn_Yes'].sort_values().iloc[1:-1]


plt.figure(figsize=(10,4),dpi=200)
sns.barplot(x=corr_df['Churn_Yes'].sort_values().iloc[1:-1].index,y=corr_df['Churn_Yes'].sort_values().iloc[1:-1].values)
plt.title("Feature Correlation to Yes Churn")
plt.xticks(rotation=90);

df['Contract'].unique()

plt.figure(figsize=(12,6), dpi = 200)
sns.displot(data = df, x = 'tenure', bins = 50, col = 'Contract', row = 'Churn')

sns.scatterplot(data = df, x = 'TotalCharges', y = 'MonthlyCharges', hue = 'Churn')

# def cohort(tenure):
#     if tenure<13:
#         return '0-12 Months'
#     elif tenure<25:
#         return '12-24 Months'
#     elif tenure<49:
#         return '24-48 Months'
#     else:
#         return 'Over 48 months'

# df['Tenure Cohort'] = df['tenure'].apply(cohort)

plt.figure(figsize=(12,6),dpi=200)
sns.scatterplot(data=df,x='MonthlyCharges',y='TotalCharges',hue='Tenure Cohort', linewidth=0.5,alpha=0.5,palette='plasma')

plt.figure(figsize=(10,4),dpi=200)
sns.countplot(data=df,x='Tenure Cohort',hue='Churn')


#Predicting Model 
X = df.drop('Churn',axis = 1)
X = pd.get_dummies(df, drop_first=True)
y = df['Churn']

from sklearn.model_selection import train_test_split, GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.8, test_size = 0.2, random_state = 202)

X_eval, X_test, y_eval, y_test = train_test_split(X_test, y_test, train_size=0.5, test_size=0.5, random_state=202)

from sklearn.ensemble import RandomForestClassifier
forest_model = RandomForestClassifier()

param_grid = {'n_estimators':[5,10,15,30,40,50,60,70,80,90,100], 'criterion':['gini', 'entropy']}
grid_model = GridSearchCV(forest_model, param_grid, cv = 10, verbose = 2)
grid_model.fit(X_train, y_train)

grid_model.best_estimator_
grid_model.best_params_

predicted = grid_model.predict(X_test)
new_predict = grid_model.predict(X_eval)

from sklearn.metrics import plot_confusion_matrix, classification_report
plot_confusion_matrix(grid_model,X_test)
print(classification_report(y_eval, new_predict))
print(classification_report(y_test, predicted))


