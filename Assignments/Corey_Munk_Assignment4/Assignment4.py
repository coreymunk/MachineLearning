#%%
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, SGDRegressor, Lasso, ElasticNet, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.metrics import make_scorer, mean_squared_error
#%%
# Part 1 - Data Exploration
# Data README - https://github.com/arjayit/cs4432_data/blob/master/bike_share_Readme.txt

# Load data
url = 'https://raw.githubusercontent.com/arjayit/cs4432_data/master/bike_share_hour.csv'
df = pd.read_csv(url)

# Convert categorical columns to pandas cetegory type
cat_cols = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
df[cat_cols] = df[cat_cols].astype('category')

#%% There appears to be no null values in the dataset
df.info()

#%% Descriptive analytics on numeric columns
df.describe().T

#%% Show months in each season
# It seems that the seasons dont fully align with expected months
seasons_dict = {1:'spring', 2:'summer', 3:'fall', 4:'winter'}
months_by_season = df.groupby('season')['mnth'].unique()

for season_id, season_name in seasons_dict.items():
    months = months_by_season[season_id]
    print(f'{season_name}: {months.tolist()}')

# Barplot showing ride count by season. 
# "Fall" (June, July, Aug, Sept) has the highest ride count while "Spring" (Dec, Jan, Feb, March) has the lowest.
sns.barplot(data=df, x='season', y='cnt', estimator=sum, errorbar=None)
plt.title('Ride Count by Season')
plt.xlabel('Season')
plt.ylabel('Ride Count')
plt.show()

# Barplot showing ride count by working day. 
# Fairly even distribution with workdays having more rides.
sns.barplot(data=df, x='workingday', y='cnt', estimator=sum, errorbar=None)
plt.title('Ride Count by Workday')
plt.xlabel('Workday')
plt.ylabel('Ride Count')
plt.show()

# Barplot showing ride count by month. 
# It looks like June, August, and September have the highest ride counts.
sns.barplot(data=df, x='mnth', y='cnt', estimator=sum, errorbar=None)
plt.title('Ride Count by Month')
plt.xlabel('Month')
plt.ylabel('Ride Count')
plt.show()

# Barplot showing ride count by weather. 
# Group 4 (Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog) has the fewest rides.
sns.barplot(data=df, x='weathersit', y='cnt', estimator=sum, errorbar=None)
plt.title('Ride Count by Weather')
plt.xlabel('Weather')
plt.ylabel('Ride Count')
plt.show()

# Point plot showing ride count by weather & season. 
# It seems that ride count decreases within each season as weather becomes more inclement. 
sns.pointplot(data=df, x='weathersit', y='cnt', estimator=sum, hue='season', errorbar=None)
plt.title('Ride Count Weather & Season ')
plt.xlabel('Weather')
plt.ylabel('Ride Count')
plt.show()

# Barplot showing ride count by hour. 
# There are spikes in ride count around 8am and 5pm.
sns.barplot(data=df, x='hr', y='cnt', estimator=sum, errorbar=None)
plt.title('Ride Count by Hour')
plt.xlabel('Hour')
plt.ylabel('Ride Count')
plt.show()

# Barplot showing ride count by hour on non work days. 
# There is a normal distribution around 1pm.
df_nonworkdays = df[(df['workingday']==0)]
sns.barplot(data=df_nonworkdays, x='hr', y='cnt', estimator=sum, errorbar=None)
plt.title('Ride Count by Hour (Non Working Days)')
plt.xlabel('Hour')
plt.ylabel('Ride Count')
plt.show()

#%% 
# Part 2 - Data Preparation

# Correlation Matrix
# We can see high positive correlation between count and casual/registered features which is expected. 
# Additionally, we see a moderate positive correlation between count and temp/atemp while we see a 
# moderate negative corelation between count and humidity. These observations all make sense.
corr_matrix = df.corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Scale numeric features
numeric_features = df.select_dtypes(include=['int64', 'float64'])
scaled_numeric_features = StandardScaler().fit_transform(numeric_features)
scaled_numeric_features_df = pd.DataFrame(scaled_numeric_features, columns=numeric_features.columns)
df[numeric_features.columns] = scaled_numeric_features_df

# Drop some columns
df = df.drop(columns=['casual', 'registered', 'dteday', 'instant'])

# Show distribution of cnt column
# The scaled values appear to be right skewed
plt.hist(df['cnt'], bins=20)
plt.title('Distribution of cnt')
plt.show()

# Split data into train/test set
X = df.drop(columns='cnt')
y = df['cnt']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=321)

# Fit a linear regression model and get baseline metrics
model_LR = LinearRegression()
scoring = {'r2': 'r2', 'mse': make_scorer(mean_squared_error)}
cv_LR = cross_validate(model_LR, X_train, y_train, scoring=scoring)

r2 = cv_LR['test_r2']
mse = cv_LR['test_mse']
rmse = np.sqrt(mse)

print('**Baseline Linear Regression Model**')
print('r2: ', np.mean(r2))
print('mse: ', np.mean(mse))
print('rmse: ', np.mean(rmse))

#%%
# Part 3 - Model Training

# Create dummy columns for categorical data
categorical_cols = df.select_dtypes(include='category').columns
df = pd.get_dummies(df, columns = categorical_cols, prefix = categorical_cols)

# Split data into new train/test set
X = df.drop(columns='cnt')
y = df['cnt']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=321)

# Fit another linear regression model and get baseline metrics
model_LR2 = LinearRegression()
scoring = {'r2': 'r2', 'mse': make_scorer(mean_squared_error)}
cv_LR2 = cross_validate(model_LR2, X_train, y_train, scoring=scoring)

r2 = cv_LR2['test_r2']
mse = cv_LR2['test_mse']
rmse = np.sqrt(mse)

print('**New Linear Regression Model**')
print('r2: ', np.mean(r2))
print('mse: ', np.mean(mse))
print('rmse: ', np.mean(rmse))

#%%
# Fit Various Model Types
models = [
    ('Descision Tree', DecisionTreeRegressor(random_state=0)),
    ('Random Forest', RandomForestRegressor(random_state=0, n_estimators=30)),
    ('SGD', SGDRegressor(max_iter=1000, tol=1e-3)),
    ('Lasso', Lasso(alpha=.1)),
    ('Elastic Net', ElasticNet(random_state=0)),
    ('Ridge', Ridge(alpha=.5)),
    ('Bagging', BaggingRegressor())]

results_df = pd.DataFrame(columns=['model', 'r2', 'mse', 'rmse'])
scoring = {'r2': 'r2', 'mse': make_scorer(mean_squared_error)}

for model_name, model in models:
    cv = cross_validate(model, X_train, y_train, scoring=scoring)
    r2 = cv['test_r2']
    mse = cv['test_mse']
    rmse = np.sqrt(mse)

    model_results = pd.DataFrame({
        'model': [model_name], 
        'r2': [np.mean(r2)], 
        'mse': [np.mean(mse)], 
        'rmse': [np.mean(rmse)]
    })

    results_df = pd.concat([results_df, model_results], ignore_index = True)

results_df.sort_values(by='rmse', ascending=True)

#%%
#Part 4 - Model Tuning

# Define parameters for search
params = {
    'bootstrap': [True, False],
    'max_depth': np.arange(10,111, 11),
    'max_features': ['auto', 'sqrt'], #{'sqrt', 'log2'}
    'min_samples_split': [2,5,10],
    'min_samples_leaf': [1,2,4],
    'n_estimators': np.arange(200,2001, 10)
}

scoring = {'r2': 'r2', 'mse': make_scorer(mean_squared_error)}

# Instatiate Random Forrest Model
model_RF = RandomForestRegressor(random_state=0)

# Define search variables
rand_search = RandomizedSearchCV(model_RF, 
                                 param_distributions = params,
                                 n_iter = 20,
                                 scoring = scoring,
                                 cv = 3,
                                 n_jobs = os.cpu_count() - 1,
                                 refit='r2',
                                 random_state= 321)

# Fit randomized model 
rand_search.fit(X_train, y_train)
best_model = rand_search.best_estimator_
best_params = rand_search.best_params_
print('**Randomized Search Results**')
print('Best Model: ', best_model,'\n')
print('Best Params: ', best_params,'\n')

# Best model cross validation results
cv = cross_validate(best_model, X_train, y_train, scoring = scoring, cv = 3)
r2 = cv['test_r2']
mse = cv['test_mse']
rmse = np.sqrt(mse)

print('**Best Model Cross Validation Results**')
print('r2: ', np.mean(r2))
print('mse: ', np.mean(mse))
print('rmse: ', np.mean(rmse),'\n')

# Run predictions using the best model
y_pred = best_model.predict(X_test)

r2_test = best_model.score(X_test, y_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))

print('**Best Model Test Results**')
print('r2: ', np.mean(r2_test))
print('rmse: ', np.mean(rmse_test))

#%%