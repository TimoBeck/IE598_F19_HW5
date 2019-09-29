import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn import svm
from sklearn import metrics
import scipy.stats as stats
from sklearn.metrics import mean_squared_error

df = pd.read_csv("hw5_treasury yield curve data.csv")

#Part 1 - EDA
df = df.dropna()
print('Summary statistic of our dataset')
print(df.describe())
print('Head of our dataset')
print(df.head())
print('Heat map of dataset:')
corMat = pd.DataFrame(df.corr())
plt.pcolor(corMat)
plt.show()

#Linear Regression
X, y = df.iloc[:,1:31].values, df.iloc[:, 31].values
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)
y_test_pred = reg.predict(X_test)
y_train_pred = reg.predict(X_train)
print('R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))

# SVR Regressor
clf_svr = svm.SVR(kernel='linear')
clf_svr.fit(X_train,y_train)
y_pred_train_SVM = clf_svr.predict(X_train)
y_pred_test_SVM = clf_svr.predict(X_test)
print('R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_pred_train_SVM),r2_score(y_test, y_pred_test_SVM)))

#Part 2 - PCA
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

# Compute and display the explained variance ratio for all components 
pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
print(pca.explained_variance_ratio_)
plt.bar(range(1,31), pca.explained_variance_ratio_, alpha=0.5, align='center',label='individual explained variance')
cum_var = np.cumsum(pca.explained_variance_ratio_)
plt.step(range(1,31), cum_var, where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()

# Compute and display the explained variance ratio for n_components = 3
pca_2 = PCA(n_components=3)
X_train_pca_n3 = pca_2.fit_transform(X_train_std)
features = range(pca_2.n_components_)
plt.bar(features, pca_2.explained_variance_ratio_)
plt.xticks(features)
plt.ylabel('variance')
plt.xlabel('PCA feature')
plt.show()

# Fit a linear regression after PCA with n = 3
X_test_pca_n3 = pca_2.transform(X_test_std)
reg.fit(X_train_pca_n3,y_train)
y_test_pred = reg.predict(X_test_pca_n3)
y_train_pred = reg.predict(X_train_pca_n3)
print('R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),mean_squared_error(y_test, y_test_pred)))

# SVR Regressor after PCA with n = 3
clf_svr = svm.SVR(kernel='linear')
clf_svr.fit(X_train_pca_n3,y_train)
y_pred_train_SVM = clf_svr.predict(X_train_pca_n3)
y_pred_test_SVM = clf_svr.predict(X_test_pca_n3)
print('R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_pred_train_SVM),r2_score(y_test, y_pred_test_SVM)))
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_pred_train_SVM),mean_squared_error(y_test, y_pred_test_SVM)))

print("My name is Timothee Becker")
print("My NetID is: tbecker5")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
