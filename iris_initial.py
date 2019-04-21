from sklearn.datasets import load_iris
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

dataset = load_iris()
#conversion of sklearn dataset to pandas
ds = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y =dataset.target
print(y)
ds['class'] = y

	
# shape
print(ds.shape)

# peeking at the start of the dataset
print(ds.head())

#describing the loaded dataset
print(ds.describe())

#classwise distribution
#print(dataset.groupby('class').size())

#boxplot
#ds.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show
#histograms
ds.hist()
plt.show()
	
# scatter plot matrix
scatter_matrix(ds)
plt.show()


	
# Split-out validation dataset
array = ds.values
X = array[:,0:4]
print(X)
Y = array[:,4]
print(Y)

#validation size refers to the testing datasets. here 20% goes to testing and 80% to train
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []

#This will split our dataset into 10 parts, train on 9 and test on 1 and repeat for all combinations of train-test splits.

for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

