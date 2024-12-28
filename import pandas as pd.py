import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#to load the dataset
data = pd.read_csv('titanic.csv')
#to view the dataset
print(data.head())

#check for data types
print(data.info())
# for missing values
print(data.isnull().sum())
#summary
print(data.describe())

#data cleaning by droping unnecessary columns
#by handleing missing values
data.drop(columns=['Cabin'], inplace=True)
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

#data analysis by age
sns.histplot(data['Age'], bins=20, kde=True, color='red')
plt.title('Age Distribution')
plt.show()

#data analysis of survival rate by passenger class
sns.barplot(x='Pclass', y='Survived',color='green', data=data)
plt.title('Survival by Passenger Class')
plt.show()

#data analysis of survival rate by gender
sns.barplot(x='Sex', y='Survived',palette={'male': 'blue', 'female': 'pink'}, data=data)
plt.title('Survival by Gender')
plt.show()

#data analysis of survival rate by both pclass and gender
sns.catplot(x='Pclass', y='Survived', hue='Sex', kind='bar', data=data)
plt.title('Survival by Class and Gender')
plt.show()
