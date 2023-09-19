#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#%% load diabetes dataset
diabetes = load_diabetes()
diabetes_df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
diabetes_df['target'] = diabetes.target

#%% explore dataset
print(diabetes.keys(), '\n')
print(diabetes_df.head())
print(diabetes['DESCR'], '\n')

#%% create numpy arrays for a single feature and the target
Feature = diabetes.data[:,2] # I chose bmi since obesity increases risk for diabetes
Target = diabetes.target

#%% split data into training and test sets
Feature_train = Feature[:-20].reshape(-1,1) # up to last 20 reshaped into 2D array
Feature_test = Feature[-20:].reshape(-1,1) # last 20 reshaped into 2D array

Target_train = Target[:-20] # up to last 20
Target_test = Target[-20:] # last 20

#%% train model
lm = LinearRegression()
lm.fit(Feature_train, Target_train)

#%% use linear regression model to predict target
Target_predicted = lm.predict(Feature_test)
MeanSquaredError = mean_squared_error(Target_test, Target_predicted)
FeatureCoeff = lm.coef_

print('The first ten predictions are:\n',Target_predicted[:10]) # first ten predictions
print('The BMI feature coefficient is:',FeatureCoeff)
print('The linear model RSME is:',MeanSquaredError)

#%% visualize results
plt.scatter(Feature_test, 
            Target_test, 
            color='grey')

plt.plot(Feature_test, 
         Target_predicted,
         color='black')

plt.xlabel('BMI')
plt.ylabel('Progression')
plt.title('BMI vs Progression')

plt.show()