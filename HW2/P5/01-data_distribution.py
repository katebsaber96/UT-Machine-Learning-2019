import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import preprocessing

# Importing the data
df_Xtrain = pd.read_csv("./data/Xtrain.csv")
df_Xtest = pd.read_csv("./data/Xtest.csv")
df_Xtrain.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
df_Xtest.drop(columns=['Unnamed: 0'], axis=1, inplace=True)

# Data normalization (NVP column needs normalization)
nvp = df_Xtrain['NVP'].values.astype(float)
min_max_scalar = preprocessing.MinMaxScaler()
nvp_scaled = min_max_scalar.fit_transform(nvp.reshape(-1, 1))
df_normalized = pd.DataFrame(nvp_scaled)
df_Xtrain['NVP'] = df_normalized

# Initial Visualization od the data
print(df_Xtrain.head(20))

# Plotting data distribution over NVP
plt.figure(figsize=(15, 10))
plt.tight_layout()
sns.distplot(df_Xtrain['NVP'])
# plt.show()
plt.savefig('./plots/mean_NVP.png')