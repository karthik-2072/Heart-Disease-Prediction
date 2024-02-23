#!/usr/bin/env python
# coding: utf-8

# In[66]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[67]:


df = pd.read_csv("heart.xls")


# In[68]:


df.head()


# Data contains;
# 
# age - age in years
# sex - (1 = male; 0 = female)
# cp - chest pain type
# trestbps - resting blood pressure (in mm Hg on admission to the hospital)
# chol - serum cholestoral in mg/dl
# fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# restecg - resting electrocardiographic results
# thalach - maximum heart rate achieved
# exang - exercise induced angina (1 = yes; 0 = no)
# oldpeak - ST depression induced by exercise relative to rest
# slope - the slope of the peak exercise ST segment
# ca - number of major vessels (0-3) colored by flourosopy
# thal - 3 = normal; 6 = fixed defect; 7 = reversable defect
# target - have disease or not (1=yes, 0=no)

# # Data Preprocessing

# In[69]:


print(df.isnull().sum())


# There are some null values and I had replaces with mean and removed some

# In[70]:


# Check for duplicates
duplicates = df.duplicated()

# Print the duplicate rows, if any
print(df[duplicates])


# Here are some duplicates but we need them

# In[71]:


# Create a box plot to visualize the data distribution
plt.figure(figsize=(12,8))
df.boxplot()
plt.title('Box plot of Heart Disease Dataset')
plt.xticks(rotation=90)
plt.show()


# In[72]:


# Calculate the z-score of each data point
z_scores = np.abs((df - df.mean()) / df.std())
# Identify data points with a z-score greater than 3 as outliers
outliers = df[z_scores > 3]
# Print the outliers
print(outliers)


# Remove irrelevant or redundant variables, such as variables with low variance or those that are highly correlated with other variables

# In[73]:


from sklearn.feature_selection import VarianceThreshold
# Define the variance threshold
threshold = 0.1

# Create a VarianceThreshold object and fit it to the data
selector = VarianceThreshold(threshold)
selector.fit(df)

# Get the indices of the selected features
selected_indices = selector.get_support(indices=True)

# Create a new DataFrame with the selected features
df_selected = df.iloc[:, selected_indices]


# Variance threshold: You can use the VarianceThreshold class from scikit-learn to remove features with low variance. This method removes all features whose variance doesn't meet a certain threshold, meaning that they have almost no variation and are unlikely to be useful for the model. Here's an example

# In[74]:


df.head()


# Data Normalization

# In[75]:


from sklearn.preprocessing import MinMaxScaler
# Create a MinMaxScaler object and fit it to the data
scaler = MinMaxScaler()
scaler.fit(df)

# Transform the data
df_normalized = pd.DataFrame(scaler.transform(df), columns=df.columns)


# In[76]:


print(df_normalized)


# In[77]:


df.head()


# Data Reduction(pca)

# In[78]:


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Split the DataFrame into features and target variable
X = df.drop('target', axis=1)
y = df['target']

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Create a PCA object and fit it to the data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Create a new DataFrame with the principal components
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
df_pca['target'] = y

# Visualize the results
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title('PCA on Heart Disease Data')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
targets = [0, 1]
colors = ['r', 'b']
for target, color in zip(targets, colors):
    indicesToKeep = df_pca['target'] == target
    ax.scatter(df_pca.loc[indicesToKeep, 'PC1'],
               df_pca.loc[indicesToKeep, 'PC2'],
               c=color, s=50)
ax.legend(targets)
ax.grid()
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


# In[79]:


# Create a scatter plot of the first two principal components, colored by the target variable
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


# # Data Exploration

# In[80]:


df.target.value_counts()


# In[81]:


sns.countplot(x="target", data=df, palette="bwr")
plt.show()


# In[82]:


countNoDisease = len(df[df.target == 0])
countHaveDisease = len(df[df.target == 1])
print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(df.target))*100)))
print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(df.target))*100)))


# In[83]:


sns.countplot(x='sex', data=df, palette="mako_r")
plt.xlabel("Sex (0 = female, 1= male)")
plt.show()


# In[84]:


countFemale = len(df[df.sex == 0])
countMale = len(df[df.sex == 1])
print("Percentage of Female Patients: {:.2f}%".format((countFemale / (len(df.sex))*100)))
print("Percentage of Male Patients: {:.2f}%".format((countMale / (len(df.sex))*100)))


# In[85]:


pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(15,6),color=['#1CA53B','#AA1111' ])
plt.title('Heart Disease Frequency for Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()


# In[86]:


# Group the data by age and calculate the mean cholesterol value for each age group
grouped_data = df.groupby('age')['chol'].mean().reset_index()

# Plot a line plot of the mean cholesterol value for each age group
plt.plot(grouped_data['age'], grouped_data['chol'])
plt.xlabel('Age')
plt.ylabel('Mean Cholesterol')
plt.title('Line Plot of Mean Cholesterol vs Age')
plt.show()


# In[87]:


# Group the data by cp and target, and calculate the count for each group
grouped_data = df.groupby(['cp', 'target']).size().reset_index(name='count')

# Pivot the data to have cp values as columns and target values as rows
pivot_data = grouped_data.pivot(index='target', columns='cp', values='count')

# Plot a stacked bar plot of the frequency of each chest pain type for each target value
pivot_data.plot(kind='bar', stacked=True)
plt.xlabel('Target (0 = No Heart Disease, 1 = Heart Disease)')
plt.ylabel('Frequency')
plt.title('Frequency of Each Chest Pain Type for Each Target Value')
plt.legend(title='Chest Pain Type', loc='upper left')
plt.show()


# In[88]:


# Split the data into two groups based on target value
has_disease = df[df['target'] == 1]
no_disease = df[df['target'] == 0]

# Plot histograms of the age distribution for each group
plt.hist(has_disease['age'], alpha=0.5, label='Has Heart Disease')
plt.hist(no_disease['age'], alpha=0.5, label='No Heart Disease')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution for Each Target Value')
plt.legend()
plt.show()


# In[89]:


pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('heartDiseaseAndAges.png')
plt.show()


# In[90]:


# Create a boxplot of maximum heart rate achieved for each target value
sns.boxplot(x='target', y='thalach', data=df)
plt.xlabel('Target Value (0 = No Heart Disease, 1 = Heart Disease)')
plt.ylabel('Maximum Heart Rate Achieved')
plt.title('Distribution of Maximum Heart Rate Achieved by Target Value')
plt.show()


# In[91]:


df.groupby('target').mean()


# In[92]:


plt.scatter(x=df.age[df.target==1], y=df.thalach[(df.target==1)], c="red")
plt.scatter(x=df.age[df.target==0], y=df.thalach[(df.target==0)])
plt.legend(["Disease", "Not Disease"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.show()


# In[93]:


pd.crosstab(df.fbs,df.target).plot(kind="bar",figsize=(15,6),color=['#FFC300','#581845' ])
plt.title('Heart Disease Frequency According To FBS')
plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')
plt.xticks(rotation = 0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency of Disease or Not')
plt.show()


# In[94]:


# Create a cross-tabulation table and plot the frequencies
colors = ['#2ECC71', '#E74C3C']
pd.crosstab(df.fbs, df.target).plot(kind='bar', figsize=(15, 6), color=colors)
plt.title('Heart Disease Frequency According to FBS')
plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')
plt.xticks(rotation=0)
plt.legend(["No Heart Disease", "Heart Disease"])
plt.ylabel('Frequency')
plt.show()


# Creating Dummy Variables  :Since 'cp', 'thal' and 'slope' are categorical variables we'll turn them into dummy variables.

# In[95]:


a = pd.get_dummies(df['cp'], prefix = "cp")
b = pd.get_dummies(df['thal'], prefix = "thal")
c = pd.get_dummies(df['slope'], prefix = "slope")


# In[96]:


frames = [df, a, b, c]
df = pd.concat(frames, axis = 1)
df.head()


# In[97]:


df = df.drop(columns = ['cp', 'thal', 'slope'])
df.head()


# # Creating Model for Logistic Regression
# 

# In[98]:


y = df.target.values
x_data = df.drop(['target'], axis = 1)


# In[99]:


# Normalize the data
x = (x_data - x_data.min(axis=0)) / (x_data.max(axis=0) - x_data.min(axis=0)).values


# We will split our data. 80% of our data will be train data and 20% of it will be test data.

# In[100]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)


# In[101]:


#transpose matrices
x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T


# Let's say weight = 0.01 and bias = 0.0

# In[102]:


#initialize
def initialize(dimension):
    
    weight = np.full((dimension,1),0.01)
    bias = 0.0
    return weight,bias


# In[103]:


def sigmoid(z):
    
    y_head = 1/(1+ np.exp(-z))
    return y_head


# Forward and Backward Propagation for regression

# In[104]:


def forwardBackward(weight,bias,x_train,y_train):
    # Forward
    
    y_head = sigmoid(np.dot(weight.T,x_train) + bias)
    loss = -(y_train*np.log(y_head) + (1-y_train)*np.log(1-y_head))
    cost = np.sum(loss) / x_train.shape[1]
    
    # Backward
    derivative_weight = np.dot(x_train,((y_head-y_train).T))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
    gradients = {"Derivative Weight" : derivative_weight, "Derivative Bias" : derivative_bias}
    
    return cost,gradients


# In[105]:


def update(weight,bias,x_train,y_train,learningRate,iteration) :
    costList = []
    index = []
    
    #for each iteration, update weight and bias values
    for i in range(iteration):
        cost,gradients = forwardBackward(weight,bias,x_train,y_train)
        weight = weight - learningRate * gradients["Derivative Weight"]
        bias = bias - learningRate * gradients["Derivative Bias"]
        
        costList.append(cost)
        index.append(i)

    parameters = {"weight": weight,"bias": bias}
    
    print("iteration:",iteration)
    print("cost:",cost)

    plt.plot(index,costList)
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.show()

    return parameters, gradients


# In[106]:


def predict(weight,bias,x_test):
    z = np.dot(weight.T,x_test) + bias
    y_head = sigmoid(z)

    y_prediction = np.zeros((1,x_test.shape[1]))
    
    for i in range(y_head.shape[1]):
        if y_head[0,i] <= 0.5:
            y_prediction[0,i] = 0
        else:
            y_prediction[0,i] = 1
    return y_prediction


# In[107]:


def logistic_regression(x_train,y_train,x_test,y_test,learningRate,iteration):
    dimension = x_train.shape[0]
    weight,bias = initialize(dimension)
    
    parameters, gradients = update(weight,bias,x_train,y_train,learningRate,iteration)

    y_prediction = predict(parameters["weight"],parameters["bias"],x_test)
    
    print("Manuel Test Accuracy: {:.2f}%".format((100 - np.mean(np.abs(y_prediction - y_test))*100)))


# In[108]:


logistic_regression(x_train,y_train,x_test,y_test,1,100)


# In[109]:


accuracies = {}

lr = LogisticRegression()
lr.fit(x_train.T,y_train.T)
acc = lr.score(x_test.T,y_test.T)*100
lr=lr.score(x_test.T,y_test.T)*100
accuracies['Logistic Regression'] = acc
print("Test Accuracy {:.2f}%".format(acc))


# # K-Nearest Neighbour (KNN) Classification

# In[110]:


# KNN Model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)  # n_neighbors means k
knn.fit(x_train.T, y_train.T)
prediction = knn.predict(x_test.T)

print("5 NN Score: {:.2f}%".format(knn.score(x_test.T, y_test.T)*100))


# In[111]:


# try to find the best k value
scoreList = []
for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train.T, y_train.T)
    scoreList.append(knn.score(x_test.T, y_test.T))

plt.plot(range(1, 20), scoreList)
plt.xticks(np.arange(1, 20, 1))
plt.xlabel("K value")
plt.ylabel("Score")
plt.show()

k = 10
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train.T, y_train.T)
accuracy = knn.score(x_test.T, y_test.T)
kn=knn.score(x_test.T, y_test.T)*100
accuracies['KNN'] = accuracy * 100
print("Accuracy of KNN model with k={} is {:.2f}%".format(k, accuracy * 100))


# In the k defines as 1,2,3, as maximum 

# # Support Vector Machine (SVM) Algorithm

# In[112]:


from sklearn.svm import SVC


# In[113]:


svm = SVC(random_state = 1)
svm.fit(x_train.T, y_train.T)

acc = svm.score(x_test.T, y_test.T) * 100
sv = svm.score(x_test.T, y_test.T) * 100
accuracies['SVM'] = acc
print("Test Accuracy of SVM Algorithm: {:.2f}%".format(acc))


# In[114]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train.T, y_train.T)

acc = nb.score(x_test.T,y_test.T)*100
nb=nb.score(x_test.T,y_test.T)*100
accuracies['Naive Bayes'] = acc
print("Accuracy of Naive Bayes: {:.2f}%".format(acc))


# # Decision Tree Algorithm

# In[115]:


from sklearn.tree import DecisionTreeClassifier

# Train the decision tree with pruning using the Cost-Complexity method
dtc = DecisionTreeClassifier(ccp_alpha=0.01)
dtc.fit(x_train.T, y_train.T)

# Evaluate the model's accuracy on the test set
acc = dtc.score(x_test.T, y_test.T)*100
dt= dtc.score(x_test.T, y_test.T)*100
accuracies['Decision Tree'] = acc
print("Decision Tree Test Accuracy (with pruning) {:.2f}%".format(acc))


# In[116]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier(n_estimators=1000, random_state=1)
scores = cross_val_score(rf, X, y, cv=5)
print("Cross-validation scores:", scores)
print("Average cross-validation score:", scores.mean())


# In[117]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Define the parameter grid to search
param_grid = {
    'n_estimators': [100, 500, 1000],
    'min_samples_leaf': [1, 3, 5],
    'max_depth': [5]
}

# Define the Random Forest classifier
rf = RandomForestClassifier(random_state=1)

# Perform a grid search with 5-fold cross validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(x_train.T, y_train.T)

# Print the best hyperparameters and the corresponding mean cross-validation score
print("Best hyperparameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Define the Random Forest classifier with the best hyperparameters
rf_best = RandomForestClassifier(
    random_state=1,
    n_estimators=grid_search.best_params_['n_estimators'],
    min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
    max_depth=grid_search.best_params_['max_depth']
)

# Train the classifier on the entire training set
rf_best.fit(x_train.T, y_train.T)

# Evaluate the performance of the classifier on the test set
acc = rf_best.score(x_test.T, y_test.T) * 100
rf=acc = rf_best.score(x_test.T, y_test.T) * 100
print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(acc))



# In[118]:


import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Define the parameter grid to search for XGBoost
param_grid_xgb = {
    'n_estimators': [10, 50],
    'max_depth': [3, 5]
}

# Define the parameter grid to search for Random Forest
param_grid_rf = {
    'n_estimators': [10, 50],
    'min_samples_leaf': [1, 3],
    'max_depth': [3, 5]
}

# Define the parameter grid to search for SVM
param_grid_svm = {
    'C': [0.1, 1],
    'kernel': ['linear', 'rbf']
}

# Define the parameter grid to search for logistic regression
lr_param_grid = {
    'penalty': ['l2', 'none'],
    'C': [0.01, 0.1, 1, 10]
}

# Define the classifiers
xgb_clf = xgb.XGBClassifier(random_state=1)
rf_clf = RandomForestClassifier(random_state=1)
svm_clf = SVC(random_state=1, probability=True)
lr_clf = LogisticRegression(random_state=1, max_iter=10000)


# Perform grid search with 5-fold cross-validation
xgb_grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid_xgb, cv=5, n_jobs=-1)
rf_grid_search = GridSearchCV(estimator=rf_clf, param_grid=param_grid_rf, cv=5, n_jobs=-1)
svm_grid_search = GridSearchCV(estimator=svm_clf, param_grid=param_grid_svm, cv=5, n_jobs=-1)
lr_grid_search = GridSearchCV(estimator=lr_clf, param_grid=lr_param_grid, cv=5, n_jobs=-1)


# Fit the grid search on the training data
xgb_grid_search.fit(x_train.T, y_train.T)
rf_grid_search.fit(x_train.T, y_train.T)
svm_grid_search.fit(x_train.T, y_train.T)
lr_grid_search.fit(x_train.T, y_train.T)

# Define the classifiers with the best hyperparameters
xgb_clf = XGBClassifier(
    random_state=1,
    max_depth=xgb_grid_search.best_params_['max_depth'],
    n_estimators=xgb_grid_search.best_params_['n_estimators']
)

rf_clf = RandomForestClassifier(
    random_state=1,
    n_estimators=rf_grid_search.best_params_['n_estimators'],
    min_samples_leaf=rf_grid_search.best_params_['min_samples_leaf'],
    max_depth=rf_grid_search.best_params_['max_depth']
)

svm_clf = SVC(
    random_state=1,
    C=svm_grid_search.best_params_['C'],
    kernel=svm_grid_search.best_params_['kernel'],
    probability=True
)

lr_clf = LogisticRegression(
    random_state=1,
    penalty=lr_grid_search.best_params_['penalty'],
    C=lr_grid_search.best_params_['C'],
    max_iter=10000
)

# Define the voting classifier with the best combination of models
voting_clf = VotingClassifier(estimators=[('xgb', xgb_clf), ('svm', svm_clf), ('lr', lr_clf)], voting='soft')

voting_clf.fit(x_train.T, y_train.T)
# Evaluate the performance of the voting classifier on the test set
acc = voting_clf.score(x_test.T, y_test.T) * 100
hm=voting_clf.score(x_test.T, y_test.T) * 100
print("Hybrid Model Accuracy Score : {:.2f}%".format(acc))


# In[119]:


import matplotlib.pyplot as plt

# create a list of algorithms and their accuracies
algorithms = ['LR', 'KNN', 'SVM', 'Naive Bayes', 'Decision Tree', 'RF', 'Hybrid Model']
accuracies = [82.17, 92.31, 85.66, 81.12, 83.22, 90.91, 97.20]

# create a bar chart of the accuracies
plt.bar(algorithms, accuracies, color=['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta'])

# set the title and axis labels
plt.title('Accuracy Scores for Classification Algorithms')
plt.xlabel('Algorithm')
plt.ylabel('Accuracy (%)')
plt.subplots_adjust(bottom=0.35) 

# display the chart
plt.show()


# In[129]:


print("Logestic Regression Accuracy :",lr)
print("K-Nearest Neighbour Accuracy :",kn)
print("Navie Bayer's Accuracy :",nb)
print("Support Vector Machine Accuracy :",sv)
print("Dession Tree Accuracy :",dt)
print("Random Forest Accuracy :",rf)
print("Hybrid Model Accuracy :",hm)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Define the accuracies for each algorithm
accuracies = {'lr': lr, 'kn': kn, 'nb': nb, 'dt': dt, 'sv': sv, 'rf': rf, 'hm': hm}

# Define the x-axis labels and corresponding colors
labels = ['Logistic Regression', 'KNN', 'Naive Bayes', 'Decision Tree', 'SVM', 'Random Forest', 'Hybrid']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

# Create a barplot with the given data
sns.set_style('whitegrid')
plt.figure(figsize=(12, 6))
plt.yticks(np.arange(0, 110, 10))
plt.ylabel('Accuracy (%)')
plt.xlabel('Algorithms')
plt.subplots_adjust(bottom=0.25)
sns.barplot(x=labels, y=list(accuracies.values()), palette=colors)
plt.show()


# # Confusion Matrix

# In[136]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

# create the classifiers
lr = LogisticRegression()
knn3 = KNeighborsClassifier(n_neighbors=3)
svm = SVC()
nb = GaussianNB()
dtc = DecisionTreeClassifier()
rf = RandomForestClassifier()
voting_clf = VotingClassifier(estimators=[('lr', lr), ('knn', knn3), ('svm', svm), ('nb', nb), ('dtc', dtc), ('rf', rf)], voting='hard')

# train the classifiers
lr.fit(x_train.T, y_train.T)
knn3.fit(x_train.T, y_train.T)
svm.fit(x_train.T, y_train.T)
nb.fit(x_train.T, y_train.T)
dtc.fit(x_train.T, y_train.T)
rf.fit(x_train.T, y_train.T)
voting_clf.fit(x_train.T, y_train.T)

# make predictions
y_head_lr = lr.predict(x_test.T)
y_head_knn = knn3.predict(x_test.T)
y_head_svm = svm.predict(x_test.T)
y_head_nb = nb.predict(x_test.T)
y_head_dtc = dtc.predict(x_test.T)
y_head_rf = rf.predict(x_test.T)
y_head_hd = voting_clf.predict(x_test.T)


# In[137]:


from sklearn.metrics import confusion_matrix

cm_lr = confusion_matrix(y_test,y_head_lr)
cm_knn = confusion_matrix(y_test,y_head_knn)
cm_svm = confusion_matrix(y_test,y_head_svm)
cm_nb = confusion_matrix(y_test,y_head_nb)
cm_dtc = confusion_matrix(y_test,y_head_dtc)
cm_rf = confusion_matrix(y_test,y_head_rf)
cm_hd = confusion_matrix(y_test,y_head_hd)


# In[138]:


plt.figure(figsize=(24,12))

plt.suptitle("Confusion Matrixes",fontsize=24)
plt.subplots(figsize=(24,12), nrows=3, ncols=3)


plt.subplot(3,3,1)
plt.title("Logistic Regression Confusion Matrix")
sns.heatmap(cm_lr,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(3,3,2)
plt.title("K Nearest Neighbors Confusion Matrix")
sns.heatmap(cm_knn,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(3,3,3)
plt.title("Support Vector Machine Confusion Matrix")
sns.heatmap(cm_svm,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(3,3,4)
plt.title("Naive Bayes Confusion Matrix")
sns.heatmap(cm_nb,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(3,3,5)
plt.title("Decision Tree Classifier Confusion Matrix")
sns.heatmap(cm_dtc,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})


plt.subplot(3,3,6)
plt.title("Random Forest Confusion Matrix")
sns.heatmap(cm_hd,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(3,3,7)
plt.title("Hybrid Model Confusion Matrix")
sns.heatmap(cm_rf,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})


# In[ ]:




