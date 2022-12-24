import pandas
from sklearn import linear_model
import matplotlib.pyplot as plt

"""
logistic regression to predict whether an applicant is likely to get hired or rejected. 
"""

# Load the data from the file 'LogisticRegressionData.csv' 
data = pandas.read_csv("LogisticRegressionData.csv", header = None, names=['Score1', 'Score2', 'y']) 

# Seperate the data features (score1 and Score2) from the class attribute 
X = data[['Score1','Score2']].values 
y = data['y'].values 

# Plot the data using a scatter plot to visualize the data. 
# Represent the instances with different markers of different colors based on the class labels.
m = ['o', 'x']
c = ['hotpink', '#88c999']
fig = plt.figure()
for i in range(len(data)):
    plt.scatter([data['Score1'][i]], [data['Score2'][i]], marker=m [data['y'][i]], color=c [data['y'][i]]) 
fig.canvas.draw()

# Train a logistic regression classifier to predict the class labels y using the features X
regS = linear_model.LogisticRegression() 
regS.fit(X, y) 

# Use the trained classifier on the training data to predict the class labels
y_pred = regS.predict(X) 
# visualize the classification error on the training instances,using predicted class labels
m = ['o', 'x']
c = ['red', 'blue'] 
fig = plt.figure()
for i in range(len(data)):
    plt.scatter([data['Score1'][i]], [data['Score2'][i]], marker=m [y_pred[i]], color = c [y_pred[i]])
fig.canvas.draw()
