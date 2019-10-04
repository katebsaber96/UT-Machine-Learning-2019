import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing, metrics
from sklearn.linear_model import LinearRegression, Ridge, Lasso

ridge_alpha = 100  # Ridge regression parameter
lasso_alpha = 0.001  # Lasso regression parameter


# Normalize a column
def norm(df: pd.DataFrame, column_to_normalize: str):
    temp_col = df[column_to_normalize].values.astype(float)
    min_max_scalar = preprocessing.MinMaxScaler()
    temp_col_scaled = min_max_scalar.fit_transform(temp_col.reshape(-1, 1))
    temp_df = pd.DataFrame(temp_col_scaled)
    df[column_to_normalize] = temp_df
    return df


# Importing the data
df_Xtrain = pd.read_csv("./data/Xtrain.csv")
df_Xtest = pd.read_csv("./data/Xtest.csv")
df_Xtrain.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
df_Xtest.drop(columns=['Unnamed: 0'], axis=1, inplace=True)

# Data normalization (NVP column needs normalization)
df_Xtrain = norm(df_Xtrain, 'NVP')
df_Xtest = norm(df_Xtest, 'NVP')

# Show top 10 rows of dataframe
# print(df_Xtrain.head(10))

# Dataset description
# print(df_Xtrain.describe())

# Split X columns and Y column
X_train = df_Xtrain.drop(columns=['NVP']).values
Y_train = df_Xtrain['NVP']
X_test = df_Xtest.drop(columns=['NVP']).values
Y_test = df_Xtest['NVP']
print('X_train shape: {} and Y_train shape: {}'.format(X_train.shape, Y_train.shape))
print('X_test shape: {} and Y_test shape: {}'.format(X_test.shape, Y_test.shape))

# training part for Multivariate linear regression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Find coefficients
coeff_df = pd.DataFrame(regressor.coef_, list(df_Xtrain.columns)[1:], columns=['Coefficient'])
# print(coeff_df)

# Prediction matters!
Y_pred = regressor.predict(X_test)

# Analysis of predictions
df_compare = pd.DataFrame({'Actual': Y_test, 'Multivariate_Regression': Y_pred})
# print(df_compare.head(20))

# Errors and score
print('\n', 'Multivariate linear regression model', '\n')
print('Number of used coefficients:', np.sum(regressor.coef_ != 0))
print('R squared Score:', regressor.score(X_test, Y_test))
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))

# Training part for Ridge regression
rr = Ridge(alpha=ridge_alpha)
rr.fit(X_train, Y_train)

# Ridge model evaluation
Y_pred_ridge = rr.predict(X_test)
df_compare['ridge'] = Y_pred_ridge

# Errors and score
print('\n', 'Ridge regression model', '\n')
print('Number of used coefficients:', np.sum(rr.coef_ != 0))
print('R squared test Score:', rr.score(X_test, Y_test))
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred_ridge))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred_ridge))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_ridge)))

# Training part for Lasso
lasso = Lasso(alpha=lasso_alpha, max_iter=10e5)
lasso.fit(X_train, Y_train)

# Ridge model evaluation
Y_pred_lasso = lasso.predict(X_test)
df_compare['lasso'] = Y_pred_lasso

# Errors and score
print('\n', 'Lasso regression model', '\n')
print('Number of used coefficients:', np.sum(lasso.coef_ != 0))
print('R squared test Score:', lasso.score(X_test, Y_test))
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred_lasso))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred_lasso))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_lasso)))

# Plot the differences
df_compare.sample(25).plot(kind='bar', figsize=(10, 8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
# plt.show()
plt.savefig('./plots/actual_predicted_difference.png')
