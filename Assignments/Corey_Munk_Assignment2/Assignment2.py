#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score

#%% load and print description
diabetes = load_diabetes()
print(diabetes['DESCR'], '\n')

#%% create df, observe data types, and determine if there is missing data
diabetes_df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
diabetes_df['target'] = diabetes.target

print(diabetes_df.info())

#%% perform descriptive statistics on numeric columns
print(diabetes_df.describe().T)

#%% show histograms of each columns distribtion
diabetes_df.hist(bins=20, figsize=(10,10))

#%% split data for train and test sets
X = diabetes_df.drop('target', axis=1)
y = diabetes_df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=321)

#%% correlation matrix
correlation_matrix = diabetes_df.corr()

fig = plt.figure(figsize= (8, 6))
sns.heatmap(correlation_matrix.round(2), annot=True, vmin=-1, vmax=1)
plt.show()

# It looks like bmi, s5, and bp are all fairly highly corelated to the target variable.

#%% pairplot of top correlated features
cols = diabetes_df[['bmi', 's5', 'bp', 'target']]

sns.pairplot(cols, kind='scatter') 
plt.show()

#%% train model and print RMSE
lm = LinearRegression()
lm.fit(X_train, y_train)

y_pred = lm.predict(X_train)
mse = mean_squared_error(y_train, y_pred)
rmse = np.sqrt(mse)

print('*Linear Regression*')
print(f'RMSE: {rmse}')

#%% cross_val_score for tree regressor and random forest regressor
dt = DecisionTreeRegressor(random_state=321)
dt_cvs = cross_val_score(estimator = dt, 
                         X = X_train, 
                         y = y_train, 
                         scoring = 'neg_mean_squared_error') #CVS maximizes score negative mse for scoring
dt_rmse = np.sqrt(-dt_cvs)

print('*Decision Tree Regressor*')
print(f'Mean RMSE: {np.mean(dt_rmse)}')
print(f'Std Dev: {np.std(dt_rmse)}\n')

rf = RandomForestRegressor(random_state=321)
rf_cvs = cross_val_score(estimator = rf,
                         X = X_train,
                         y = y_train, 
                         scoring = 'neg_mean_squared_error') #CVS maximizes score negative mse for scoring
rf_rmse = np.sqrt(-rf_cvs)

print('*Random Forest Regressor*')
print(f'Mean RMSE: {np.mean(rf_rmse)}')
print(f'Std Dev: {np.std(rf_rmse)}')

# It appears that the untuned Random Forest model performed better than the untuned Decision Tree.

#%% grid search cross validation
print('*Random Forest Params*')
print(f'n_estimators: {rf.n_estimators}')
print(f'max_features: {rf.max_features}')
print(f'max_depth: {rf.max_depth}\n')

rf_tuned = RandomForestRegressor(random_state=321)

## Grid Search 1
rf_grid1 = {'n_estimators': [3,10,30],
           'max_features': [2,4,6,8]}

rf_grid_search1 = GridSearchCV(rf_tuned,
                              rf_grid1,
                            #   verbose = 5,
                              scoring = 'neg_mean_squared_error')

rf_grid_search1.fit(X_train, y_train)
#best model
rf_tuned1 = rf_grid_search1.best_estimator_
#rmse
rf_tuned1_cvr = pd.DataFrame(rf_grid_search1.cv_results_)
rf_tuned1_cvr['rmse'] = np.sqrt(-rf_tuned1_cvr['mean_test_score'])
rf_tuned1_cvr = rf_tuned1_cvr.sort_values(by='rmse')
#feature importances
rf_tuned1_featureimportance = rf_tuned1.feature_importances_
rf1_fi = pd.DataFrame({'Feature': X_train.columns, 'Importance': rf_tuned1_featureimportance})

print('***Tuning Round 1***\n')
print(f'Best Params: {rf_grid_search1.best_params_}\n')
print(f'Best Model: {rf_tuned1}\n')
print(f'RMSE Comparison:\n {rf_tuned1_cvr[["rmse","params"]]}\n')
print(f'Feature Importance:\n {rf1_fi.sort_values(by="Importance", ascending=False)}\n')

## Grid Search 2
rf_grid2 = {'n_estimators': [3,10],
           'max_features': [2,3,4],
           'bootstrap': [False]}

rf_grid_search2 = GridSearchCV(rf_tuned,
                              rf_grid2,
                            #   verbose = 5,
                              scoring = 'neg_mean_squared_error')

rf_grid_search2.fit(X_train, y_train)
#best model
rf_tuned2 = rf_grid_search2.best_estimator_ 
#rmse
rf_tuned2_cvr = pd.DataFrame(rf_grid_search2.cv_results_)
rf_tuned2_cvr['rmse'] = np.sqrt(-rf_tuned2_cvr['mean_test_score'])
rf_tuned2_cvr = rf_tuned2_cvr.sort_values(by='rmse')
#feature importances
rf_tuned2_featureimportance = rf_tuned2.feature_importances_
rf2_fi = pd.DataFrame({'Feature': X_train.columns, 'Importance': rf_tuned2_featureimportance})

print('***Tuning Round 2***\n')
print(f'Best Params: {rf_grid_search2.best_params_}\n')
print(f'Best Model: {rf_tuned2}\n')
print(f'RMSE Comparison:\n {rf_tuned2_cvr[["rmse","params"]]} \n')
print(f'Feature Importance:\n {rf1_fi.sort_values(by="Importance", ascending=False)}')

# The feature importance seems to summarize what the correlation matrix tells us by ranking the features in the model.

#%% model evaluation

# Document the best-performing model between the single feature model you trained in 
# Assignment 1, and the models you trained in part 2 and 3 of this assignment. Evaluate the 
# best performing model against your test set. Save your model for future use