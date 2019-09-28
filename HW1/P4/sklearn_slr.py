import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load data
df = pd.read_csv('./data/data_hw1.csv')
df = df.sort_values(by=['R'], ascending=False)
# print(df.head(5))
print(df.describe())

# Initial data visualization
plt.scatter(df['R'], df['T'])
plt.ylabel('Time of recurrence')
plt.xlabel('Cell size')
plt.savefig('./plot/init_df.png')
plt.clf()

# R distribution
sns.distplot(df['R'])
plt.savefig('./plot/R_distribution.png')
plt.clf()

# T distribution
sns.distplot(df['T'])
plt.savefig('./plot/T_distribution.png')
plt.clf()

# Simple linear regression using sklearn

# Create and Reshape input and output vectors
X = df['R'].values.reshape(-1, 1)
y = df['T'].values.reshape(-1, 1)

# Split data for 90% train and 10% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Train
r = LinearRegression()
r.fit(X_train, y_train)

# Retrieve beta_0 and beta_1
beta_0, beta_1 = r.intercept_, r.coef_
print("beta_0 = {} and beta_1 = {}".format(beta_0, beta_1))

# Show output line

# Generate input
x = np.arange(start=10, stop=30)

# Plot initial data
plt.scatter(df['R'], df['T'])
plt.ylabel('Time of recurrence')
plt.xlabel('Cell size')

# Plot prediction line
plt.plot(x, (beta_0 + beta_1 * x).reshape(-1), 'r-')

# Prettify
# Regression equations.
plt.text(21.5, 18, 'y={:.2f}+{:.2f}*x'.format(float(beta_0), float(beta_1)), color='red', size=12)

# Save and show it all
plt.savefig('./plot/sklearn_output.png')
plt.show()

# Calculate error
y_pred = r.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# with open('./stat_{}.txt'.format('sklearn'), 'a') as f:
#     f.write(str(metrics.mean_absolute_error(y_test, y_pred))
#             + ', ' + str(metrics.mean_squared_error(y_test, y_pred))
#             + ', ' + str(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#             + '\n')
