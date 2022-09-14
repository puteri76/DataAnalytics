import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=10, centers=2,
                       cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50);



import seaborn as sns
sns.scatterplot(x=X[:, 0], y=X[:, 1], s=50, hue=y_true);

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.2);

"""### Examples:"""

from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape

digits.data

kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)
kmeans.cluster_centers_.shape

from scipy.stats import mode

labels = np.zeros_like(clusters)
for i in range(10):
    mask = clusters == i
    labels[mask] = mode(digits.target[mask])[0]

np.zeros_like(clusters)

mask = clusters == 1
print("which images are clustered into cluster 1?") 
print(mask)
print("")
print("What is the length of mask? ")
print(len(mask))
print("")
print("How many images are clustered into cluster 1?")
print(mask.sum())

print("Among the images that are clustered into cluster 1, what is the most possible digit?")
print("")
print(mode(digits.target[mask])[0])

from sklearn.metrics import accuracy_score
accuracy_score(digits.target, labels)

from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()  # for plot styling
mat = confusion_matrix(digits.target, labels)
sns.heatmap(mat.T, square=True, annot=True, 
            fmt='d', cbar=False,
            xticklabels=digits.target_names,
            yticklabels=digits.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label');

"""# KMeans Clustering Exercise
Download the dataset [mall_customer.csv](https://drive.google.com/file/d/1T5k0zkV8fF7-AgzY-vaqIcS6cJCdEcBc/view?usp=sharing) and perform a KMeans clustering.
"""

file = "mall_customer.csv"

import pandas as pd
df = pd.read_csv(file)

df.head()

features = ['Annual_Income_(k$)', 'Spending_Score']
X = df[features]

plt.scatter(X['Annual_Income_(k$)'], X['Spending_Score']);

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
plt.scatter(X['Annual_Income_(k$)'], X['Spending_Score'], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', s=200, alpha=0.5);

# Perform the prediction by using the trained model
step_size = 0.01

# Plot the Decision Boundaries
x_min, x_max = min(X.iloc[:,0]) - 1, max(X.iloc[:,0]) + 1
y_min, y_max = min(X.iloc[:,1]) - 1, max(X.iloc[:,1]) + 1
x_values, y_values = np.meshgrid(np.arange(x_min,x_max,step_size), np.arange(y_min,y_max,step_size))

# Predict labels for all points in the mesh
predictions = kmeans.predict(np.c_[x_values.ravel(), y_values.ravel()])
# Plot the results
predictions = predictions.reshape(x_values.shape)
plt.figure(figsize=(8,6))
plt.imshow(predictions, interpolation='nearest', extent=(x_values.min(), x_values.max(), y_values.min(), y_values.max()), 
           cmap=plt.cm.Spectral, aspect='auto', origin='lower')

plt.scatter(X.iloc[:,0],X.iloc[:,1], marker='o', facecolors='grey',edgecolors='w',s=30)
# Plot the centroids of the clusters
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:,0], centroids[:,1], marker='o', s=200, linewidths=3, 
           color='k', zorder=10, facecolors='black')

plt.title('Centroids and boundaries calculated using KMeans Clustering', fontsize=16)
plt.show()

"""## Principal Component Analysis

Prior to doing this practical, please download the [wine.csv](https://drive.google.com/file/d/1TTOhlohQlndXDErVB-BrV1xcdrzMgM-G/view?usp=sharing) dataset. Upload this dataset to current working environment. 

<br><br>
**As there are more than one thousand features, can we reduce some "less important" features?**
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#from sklearn import preprocessing
#import numpy as np
#x_array = np.array([2,3,5,6,7,4,8,7,6])
#normalized_arr = preprocessing.normalize([x_array])
#print(normalized_arr)

df_defect = pd.read_csv('wine.csv')

df_defect.shape

df_defect.describe()

df_defect.head()

X_train = pd.DataFrame(X_train,columns=X.columns)
X_train.head()
X.head()

X = df_defect.drop(['Wine'], axis=1)
y = df_defect['Wine']

print(X.shape, y.shape)
print(X.columns)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

sns.displot(x=X.Alcohol)
plt.title("Before")
plt.show()

sns.displot(x=X_train[:, 0])
plt.title("After")
plt.show()
#based on the plot, the scale changes
#before applying, the number is varied and large, 
#the number becomes smaller and close to each other after the scaler.

"""Note: Do `train_test_split()` before applying `MinMaxScaler()` or `StandardScaler()`

### Determine the number of PCs
"""

from sklearn.decomposition import PCA

pca = PCA().fit(X_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

accum_explained_var = np.cumsum(pca.explained_variance_ratio_)

min_threshold = np.argmax(accum_explained_var > 0.90) # use 90%

min_threshold

pca = PCA(n_components = min_threshold + 1)

X_train_projected= pca.fit_transform(X_train)
X_test_projected = pca.transform(X_test)

X_train_projected.shape

"""### Logistic Regression Classification without PCA"""

# Train the model

# Train the model
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report

# Logistic Regression

# without reduction
logregwithoutpca = LogisticRegression()
logregwithoutpca.fit(X_train, y_train)

logregwithoutpca_result = logregwithoutpca.predict(X_test)

print('Accuracy of Logistic Regression (without PCA) on training set: {:.2f}'
     .format(logregwithoutpca.score(X_train, y_train)))
print('Accuracy of Logistic Regression (without PCA)  on testing set: {:.2f}'
     .format(logregwithoutpca.score(X_test, y_test)))
print('\nConfusion matrix :\n',confusion_matrix(y_test, logregwithoutpca_result))
print('\n\nClassification report :\n\n', classification_report(y_test, logregwithoutpca_result))

"""### Logistic Regression Classification with PCA"""

logregwithpca = LogisticRegression()
logregwithpca.fit(X_train_projected, y_train)

logregwithpca_result = logregwithpca.predict(X_test_projected)

print('Accuracy of Logistic Regression (with PCA) on training set: {:.2f}'
     .format(logregwithpca.score(X_train_projected, y_train)))
print('Accuracy of Logistic Regression (with PCA) on testing set: {:.2f}'
     .format(logregwithpca.score(X_test_projected, y_test)))
print('\nConfusion matrix :\n',confusion_matrix(y_test, logregwithpca_result))
print('\n\nClassification report :\n\n', classification_report(y_test, logregwithpca_result))

