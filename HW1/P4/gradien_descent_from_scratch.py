import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
import random
from sys import argv


number_of_iteration = 100000


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


def bgd(beta_0, beta_1, x, y, alpha=0.0001):
    new_beta_0 = beta_0 + sum(sum(alpha * (y - beta_0 - (beta_1 * x))))
    new_beta_1 = beta_1 + sum(sum(alpha * x * (y - beta_0 - (beta_1 * x))))
    return new_beta_0, new_beta_1


def sgd(beta_0, beta_1, x, y, alpha=0.001):
    new_beta_0 = beta_0 + alpha * (y - beta_0 - (beta_1 * x))
    new_beta_1 = beta_1 + alpha * x * (y - beta_0 - (beta_1 * x))
    return new_beta_0, new_beta_1


if __name__ == '__main__':
    beta_0 = random.random()
    beta_1 = random.random()

    if argv[1] == 'bgd':
        for i in range(number_of_iteration):
            lr = 0.0001
            beta_0, beta_1 = bgd(beta_0, beta_1, X_train, y_train,lr)
            # print(beta_0, beta_1)
    elif argv[1] == 'sgd':
        for i in range(number_of_iteration):
            lr = 0.001
            rand = np.random.randint(0, len(X_train))
            beta_0, beta_1 = sgd(beta_0, beta_1, sum(X_train[rand]), sum(y_train[rand]), lr)
            # print(beta_0, beta_1)
    else:
        print('Unknown algorithm')
        exit()

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

    # Calculate error
    y_pred = beta_0 + beta_1 * X_test
    print('We used {} algorithm for {} iterations with learning rate of {}'.format(argv[1], number_of_iteration, lr))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    with open('./stat_{}.txt'.format(argv[1]), 'a') as f:
        f.write(str(metrics.mean_absolute_error(y_test, y_pred))
                + ', ' + str(metrics.mean_squared_error(y_test, y_pred))
                + ', ' + str(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
                + '\n')

    # Save and show it all
    plt.savefig('./plot/{}_output.png'.format(argv[1]))
    plt.show()

