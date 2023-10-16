#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

#%% Part 1 - Data Exploration

# Load data
data = sns.load_dataset('titanic')

# Show categorical columns and non-null counts by column
data.info()
# The 'class' and 'deck' columns are categorical

# heatmap of columns with null values
sns.heatmap(data.isnull(), cbar=False, cmap='plasma')
#%%

# Did more women or men die on the Titanic?
male_deaths = data.query("sex == 'male' and survived == 0")
print(f'The total numbero of male deaths was: {len(male_deaths)}')

female_deaths = data.query("sex == 'female' and survived == 0")
print(f'The total numbero of female deaths was: {len(female_deaths)}')

# Which passenger class was more likely to survive?
survival_by_class = data.groupby('class')['survived'].mean()
print(f'\nThe Survival Rate by Class: \n {survival_by_class}\n')

# What does the distribution of fare look like?
plt.hist(data['fare'], bins=20)
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.title('Fare Disribution')
plt.show()

# What does the distribution of non-null age values look like?
plt.hist(data['age'], bins=20)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Disribution')
plt.show()

# What is the median age of each passenger class (pclass) - show in box plot?
sns.boxplot(data=data, x='pclass', y='age')
plt.xlabel('Class')
plt.ylabel('Age')
plt.title('Age by Class')
plt.show()

#%% Part 2 - Data Cleansing

# Drop deck column
data = data.drop(columns=['deck'])

# Impute null values in age with median age for that pclass.
median_age_by_class = data.groupby('pclass')['age'].median()
data['age'] = data['age'].fillna(data['pclass'].map(median_age_by_class))

# Drop remaining rows with null values (there appears to be 2 rows w/ nulls in embarked and embark_town)
data = data.dropna()

# Show that no null values remain in the dataset
print(data.info())

print('\nFeature Reverse Engineering:')
# Reverse engineer the 'who' feature
who_conditions = [(data['age'] < 16),
                  (data['age'] >= 16) & (data['sex'] == 'male'),
                  (data['age'] >= 16) & (data['sex'] == 'female')]

who_vals = ('child', 'man', 'woman')

data['who_rev_eng'] = np.select(who_conditions, who_vals)
who_equivalence = data['who'].equals(data['who_rev_eng'])
print(f'"who_rev_eng" == "who": {who_equivalence}')

# Reverse engineer the 'adult_male' feature
adultmale_conditions = [(data['age'] >= 16) & (data['sex'] == 'male'),
                        (data['age'] < 16) & (data['sex'] == 'male'),
                        (data['age'] < 16) & (data['sex'] == 'female')]

adultmale_vals = (True, False, False)

data['adultmale_rev_eng'] = np.select(adultmale_conditions, adultmale_vals)
data['adultmale_rev_eng'] = data['adultmale_rev_eng'].astype(bool)
adultmale_equivalence = data['adult_male'].equals(data['adultmale_rev_eng'])
print(f'"adultmale_rev_eng" == "adult_male": {adultmale_equivalence}')

# Remove redundant/uneeded features
cols = ['class', 'who', 'who_rev_eng', 'adult_male', 'adultmale_rev_eng', 'alive', 'embarked']
data = data.drop(columns=cols)

# Create dummy variables for categorical columns
data = pd.get_dummies(data, columns=['sex', 'embark_town'], prefix=['sex', 'embarktown'])

# Create feature and target sets and split into train/test sets
X = data.drop(columns='survived')
y = data['survived'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=321)

#%% Part 3 - Model Training

# Logistic Regression Model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_probs = lr_model.predict_proba(X_test)[:, 1] #Select probs from positive class (2nd column).

# Support Vector Classifier
svc_model = SVC(probability=True)
svc_model.fit(X_train, y_train)
svc_pred = svc_model.predict(X_test)
svc_probs = svc_model.predict_proba(X_test)[:, 1] #Select probs from positive class (2nd column).

# Stochastic Gradient Descent Classifier
sgd_model = SGDClassifier()
sgd_model.fit(X_train, y_train)
sgd_pred = sgd_model.predict(X_test)
sgd_scores = sgd_model.decision_function(X_test)

# Classification Reports & Confusion Matricies
models = [(lr_pred, "***Linear Resgression Metrics***"), 
          (svc_pred, "***Support Vector Metrics***"), 
          (sgd_pred, "***Stochastic Gradient Descent Metrics***")]

for model_preds, model_name in models:
    print(f'{model_name}')
    print(f'{classification_report(y_test, model_preds)}')
    print(f'{confusion_matrix(y_test, model_preds)}\n')

# Receiver Operating Characteristics (ROC) plots
def generate_ROCplot(y_true, y_prob, model_name):
    false_positive_rate, treu_positive_rate, decision_thresholds = roc_curve(y_true, y_prob)
    area_under_curve = roc_auc_score(y_true, y_prob)

    plt.plot(false_positive_rate, treu_positive_rate)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve (AUC = {area_under_curve:.2f})')
    plt.show()

generate_ROCplot(y_test, lr_probs, 'Logistic Regression')
generate_ROCplot(y_test, svc_probs, 'Support Vector')
generate_ROCplot(y_test, sgd_scores, 'Stochastic Gradient Descent')

#%% Part 4 - Model Tuning

# Create SVC Pipeline
svc_pipeline = Pipeline([
    ('scalar', StandardScaler()),
    ('svc', SVC(probability=True))
])

# Fit Pipeline
svc_pipeline.fit(X_train, y_train)
svc_pred2 = svc_pipeline.predict(X_test)
svc_probs2 = svc_pipeline.predict_proba(X_test)[:, 1] #Select probs from positive class (2nd column).

# Scaled Support Vector Classifier Performance
print('***Scaled Support Vector Metrics***')
print(f'{classification_report(y_test, svc_pred2)}')
print(f'{confusion_matrix(y_test, svc_pred2)}')
generate_ROCplot(y_test, svc_probs2, 'Scaled Support Vector')

# Grid Search For SVC Pipeline

#define grid search parameters
params = {
    'svc__kernel': ['rbf'],
    'svc__gamma': [0.0001, 0.001, 0.01, 0.1, 1],
    'svc__C': [1,10,50,100,200,300],
}

#instantiate grid search model
svc_grid_search = GridSearchCV(estimator=svc_pipeline,
                           param_grid=params,
                           scoring='roc_auc')

#fit grid search model
svc_grid_search.fit(X_train, y_train)

#print best model, params, and AUC score
print('***SVC Grid Search Best Model Results***')
print(f'Best Estimator: {svc_grid_search.best_estimator_}')
print(f'Best Parameters: {svc_grid_search.best_params_}')
print(f'Best AUC: {svc_grid_search.best_score_}')

#apply best model to test data 
svc_grid_search_pred = svc_grid_search.predict(X_test)
svc_grid_search_probs = svc_grid_search.predict_proba(X_test)[:, 1] #Select probs from positive class (2nd column).

# Implement Learning Curve
train_sizes, train_scores, test_scores = learning_curve(svc_grid_search.best_estimator_,
                                                        X=X_train,
                                                        y=y_train,
                                                        scoring='roc_auc')

plt.plot(train_sizes, np.mean(train_scores, axis=1), color='red', label='Training Score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), color='green', label='Validation Score')
plt.title('learning curve')
plt.xlabel('Training Examples')
plt.ylabel('Score')
plt.legend()
plt.show()