import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv('/Users/rishabdebpaul/Documents/test code/diabetes.csv')

# printing the first 5 rows of the dataset
# diabetes_dataset.head()

# number of rows and Columns in this dataset
# diabetes_dataset.shape

# getting the statistical measures of the data
diabetes_dataset.describe()

diabetes_dataset['Outcome'].value_counts()

# 0 is nondiabetic
# 1 is diabetic

diabetes_dataset.groupby('Outcome').mean()

# separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']

print(X)

print(Y)

"""Data Standardization"""

scaler = StandardScaler()

scaler.fit(X)

standardized_data = scaler.transform(X)

print(standardized_data)

X = standardized_data
Y = diabetes_dataset['Outcome']

print(X)
print(Y)

"""Train Test Split"""

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.4, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

"""Training the Model"""

classifier = svm.SVC(kernel='linear')

#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)

"""Model Evaluation

Accuracy Score
"""

# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy)

"""Making a Predictive System"""

input_data = (8,99,84,0,0,35.4,0.388,50)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


# Selecting a subset of columns for analysis
subset_diabetes_dataset = diabetes_dataset[['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'Age', 'Outcome']]

# Pairplot
sns.pairplot(subset_diabetes_dataset, hue='Outcome', palette='husl')
plt.show()

# Distribution of Blood Pressure by Outcome
plt.figure(figsize=(10, 6))
sns.histplot(data=subset_diabetes_dataset, x='BloodPressure', hue='Outcome', kde=True, palette='husl')
plt.title('Distribution of Blood Pressure by Outcome')
plt.xlabel('Blood Pressure')
plt.ylabel('Frequency')
plt.show()

# Distribution of BMI by Outcome
plt.figure(figsize=(10, 6))
sns.histplot(data=subset_diabetes_dataset, x='BMI', hue='Outcome', kde=True, palette='husl')
plt.title('Distribution of BMI by Outcome')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.show()

# Scatterplot of Glucose vs BMI colored by Outcome
plt.figure(figsize=(10, 6))
sns.scatterplot(data=subset_diabetes_dataset, x='Glucose', y='BMI', hue='Outcome', palette='husl')
plt.title('Scatterplot of Glucose vs BMI colored by Outcome')
plt.xlabel('Glucose')
plt.ylabel('BMI')
plt.show()

# Boxplot of Age by Outcome
plt.figure(figsize=(10, 6))
sns.boxplot(data=subset_diabetes_dataset, x='Outcome', y='Age', palette='husl')
plt.title('Boxplot of Age by Outcome')
plt.xlabel('Outcome')
plt.ylabel('Age')
plt.show()
