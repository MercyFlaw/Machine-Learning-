import pandas
from sklearn import linear_model
import matplotlib.pyplot as plt

"""
linear regression to predict the profit of a restaurant, based on the number of habitants where the restaurant is located
"""

# Load the data from the file RegressionData.csv in a pandas dataframe. Make sure all the instances 
data = pandas.read_csv("RegressionData.csv", header = None, names=['X', 'y']) 
# reshape to process data
X = data['X'].values.reshape(-1,1) 
y = data['y'].values.reshape(-1,1)
# visualize the data
plt.scatter(X, y) 

# linear regression using least squares optimization
reg = linear_model.LinearRegression() 
reg.fit(X, y) 

# plot the linear fit
fig = plt.figure()
y_pred = reg.predict(X) 
plt.scatter(X,y, c='b') 
plt.plot(X, y_pred, 'r') 
fig.canvas.draw()


print("the linear relationship between X and y was modeled according to the equation: y = b_0 + X*b_1, \
where the bias parameter b_0 is equal to ", reg.intercept_,"and the weight b_1 is equal to ", reg.coef_)


# predict the profit of a restaurant, if this restaurant is located in a city of 18 habitants 
print("the profit/loss in a city with 18 habitants is ", reg.intercept_ + (18*reg.coef_))