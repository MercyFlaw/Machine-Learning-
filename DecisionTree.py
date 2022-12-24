from sklearn import datasets 
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

# load breast_cancer dataset from scikit-learn 
X, y = datasets.load_breast_cancer(return_X_y=True)

# data description
print("There are",len(X[:,1]), "instances described by", len(X[1,:]), "features.") 

# 60%=training_set 40%=testing_set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, stratify = y, random_state = 42)  
# decision tree classifier with entropy criteria to measure quality of split 
# nodes < 6 training instances are not further split
clf = tree.DecisionTreeClassifier(criterion = 'entropy', min_samples_split = 6)   
clf = clf.fit(X_train, y_train)  

# apply the decision tree 
predC = clf.predict(X_test) 

# compute the accuracy of the classifier 
print('The accuracy of the classifier is', accuracy_score(y_test, predC))  

# visualize the tree 
dtree = tree.plot_tree(clf, filled = True, fontsize=12)

# visualize the training and test error as a function of the maximum depth of the decision tree

trainAccuracy = []  
testAccuracy = [] 
# function for different depths options, ranging from 1 to 15, for the decision trees
depthOptions = 16 
for depth in range(1,depthOptions): 
    cltree = tree.DecisionTreeClassifier(criterion = 'entropy', min_samples_split = 6, max_depth = depth) 
    # Decision tree training
    cltree = cltree.fit(X_train,y_train) 
    # Training error
    y_predTrain = cltree.predict(X_train) 
    # Testing error
    y_predTest = cltree.predict(X_test) 
    # Training accuracy
    trainAccuracy.append(accuracy_score(y_train, y_predTrain)) 
    # Testing accuracy
    testAccuracy.append(accuracy_score(y_test, y_predTest)) 

# Plot of training and test accuracies vs the tree depths 
plt.plot(trainAccuracy,"bo-",testAccuracy,"ro-") 
plt.legend(['Training Accuracy','Test Accuracy']) 
plt.xlabel('Tree Depth') 
plt.ylabel('Classifier Accuracy') 
plt.show()

# GridSearchCV function to perform an exhaustive search to find the best tree depth.

parameters = {'max_depth': [*range(1,16)]} 

clf = GridSearchCV(estimator = tree.DecisionTreeClassifier(criterion = 'entropy', min_samples_split = 6, random_state = 0), param_grid = parameters) 
clf.fit(X_train, y_train) 
tree_model = clf.best_estimator_ 
print("The maximum depth of the tree should be", clf.best_params_) 

gtree = tree.plot_tree(tree_model,filled=True, fontsize = 12) 


