import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn import preprocessing

df_Xtrain = pd.read_csv("./data/Xtrain.csv")
df_Xtest = pd.read_csv("./data/Xtest.csv")
df_Xtrain.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
df_Xtest.drop(columns=['Unnamed: 0'], axis=1, inplace=True)


# Normalise NVP column
nvp = df_Xtrain['NVP'].values.astype(float)
# Create a minimum and maximum processor object
min_max_scalar = preprocessing.MinMaxScaler()
# Create an object to transform the data to fit minmax processor
nvp_scaled = min_max_scalar.fit_transform(nvp.reshape(-1, 1))
# Run the normalizer on the dataframe
df_normalized = pd.DataFrame(nvp_scaled)
df_Xtrain['NVP'] = df_normalized

X = df_Xtrain.drop(columns=['NVP'])
print(X.describe())

pca = PCA(n_components=2)
df = pd.DataFrame(pca.fit_transform(X), columns=['c1', 'c2'])


plt.scatter(df['c1'], df['c2'])
plt.xlabel('Component 1')
plt.ylabel('Component 2')
# plt.show()
plt.savefig('./plots/c1-c2.png')
