#-------------------------------------------------------------------------
# AUTHOR: Tomes, Christopher
# FILENAME: clustering.py
# SPECIFICATION: Program that uses K-Means to train on a file named training_data.csv and than tests the Homogeneity Score with testing_data.csv
# FOR: CS 4210- Assignment #5
# TIME SPENT: 30 minutes
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library

#assign your training data to X_training feature matrix

#run kmeans testing different k values from 2 until 20 clusters
     #Use:  kmeans = KMeans(n_clusters=k, random_state=0)
     #      kmeans.fit(X_training)
     #--> add your Python code

X_training = df.values
max_silhoette_coefficient = 0
silhouette_coefficients = []
for  k in range(2,21):    
     print("Testing kval ", k)
     kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
     kmeans.fit(X_training)

     #for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
     #find which k maximizes the silhouette_coefficient
     #--> add your Python code here

     temp_silhoette_coefficient = silhouette_score(X_training, kmeans.labels_)
     silhouette_coefficients.append(temp_silhoette_coefficient)
     if silhouette_score(X_training, kmeans.labels_) > temp_silhoette_coefficient:

          max_silhoette_coefficient = temp_silhoette_coefficient


#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
#--> add your Python code here
plt.plot(range(2, 21), silhouette_coefficients)
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Coefficient')
plt.title('Silhouette Coefficient vs. Number of Clusters')
plt.show()
#reading the test data (clusters) by using Pandas library
#--> add your Python code here
df = pd.read_csv('testing_data.csv', sep=',', header=None) 
#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
#--> add your Python code here
labels = np.array(df.values).reshape(1,df.shape[0])[0]
#Calculate and print the Homogeneity of this kmeans clustering
#print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
#--> add your Python code here
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())