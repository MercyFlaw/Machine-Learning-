import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer


# Load Data
data = pd.read_csv("heart_failure_clinical_records_dataset.csv")

# Verify Shape
shape = data.shape

# Drop non-applicable entries
data.dropna(inplace=True)

# Scale the data
scaler = StandardScaler()
data[['creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium']] = scaler.fit_transform(data[['creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium']])


"""
NOTE: IN ORDER TO ELBOW PLOT YELLOWBRICK EXTENSION MAY NEED TO BE INSTALLED IN PYTHON LIBRARY
"""
#-----ELBOW PLOT--------

# Elbow plot
kmeansElbow = KMeans()
visualizer = KElbowVisualizer(kmeansElbow, k=(1,10))

visualizer.fit(data)        
visualizer.show()        

# -----------------------


# K-Means clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(data[['serum_creatinine','age']])

data['set_label'] = kmeans.labels_

# Plot clusters
plt.scatter(x=data['serum_creatinine'],y=data['age'], c=data['set_label'],cmap='Set3')
plt.show()


# Describe data associated with set0
s0 = data.loc[data['set_label'] == 0]
s0Info=s0.describe()
print(s0Info)

# Describe data associated with set1
s1 = data.loc[data['set_label'] == 1]
s1Info=s1.describe()
print(s1Info)

# Describe data associated with set2
s2 = data.loc[data['set_label'] == 2]
s2Info=s2.describe()
print(s2Info)
