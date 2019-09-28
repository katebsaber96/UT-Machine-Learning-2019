import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load data
df = pd.read_csv('./data/data_hw1.csv')
df = df.sort_values(by=['R'], ascending=False)
# print(df.head(5))
# print(df.describe())

# Create and Reshape input and output vectors
X = df['R'].values.reshape(-1, 1)
y = df['T'].values.reshape(-1, 1)

# Split data for 90% train and 10% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Analytical method

# Adding necessary columns to data frame
df['RT'] = df['R'] * df['T']
df['R2'] = df['R']**2

# Analytical calculation
beta_1 = (df['RT'].mean() - (df['R'].mean() * df['T'].mean()))/(df['R2'].mean() - df['R'].mean()**2)
beta_0 = df['T'].mean() - beta_1 * df['R'].mean()
print("beta_0 = {} and beta_1 = {} - Analytical method".format(beta_0, beta_1))

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
plt.savefig('./plot/analytical_output.png')
plt.show()

# Calculate error
y_pred = beta_0 + beta_1 * X_test
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# with open('./stat_{}.txt'.format('analytical'), 'a') as f:
#     f.write(str(metrics.mean_absolute_error(y_test, y_pred))
#             + ', ' + str(metrics.mean_squared_error(y_test, y_pred))
#             + ', ' + str(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#             + '\n')