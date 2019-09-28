import pandas as pd
from matplotlib import pyplot as plt

df_bgd = pd.read_csv('stat_bgd.txt', header=None, names=['MAE', 'MSE', 'RMSE'])
df_sgd = pd.read_csv('stat_sgd.txt', header=None, names=['MAE', 'MSE', 'RMSE'])
df_sklearn = pd.read_csv('stat_sklearn.txt', header=None, names=['MAE', 'MSE', 'RMSE'])
df_analytical = pd.read_csv('stat_analytical.txt', header=None, names=['MAE', 'MSE', 'RMSE'])

mean_mae_bgd, mean_mse_bgd, mean_rmse_bgd = df_bgd['MAE'].mean(), df_bgd['MSE'].mean(), df_bgd['RMSE'].mean()
mean_mae_sgd, mean_mse_sgd, mean_rmse_sgd = df_sgd['MAE'].mean(), df_sgd['MSE'].mean(), df_sgd['RMSE'].mean()

print('analytical: MAE={} MSE={} RMSE={}'.format(float(df_analytical['MAE'].values), float(df_analytical['MSE'].values), float(df_analytical['RMSE'].values)))
print('sklearn: MAE={} MSE={} RMSE={}'.format(float(df_sklearn['MAE'].values), float(df_sklearn['MSE'].values), float(df_sklearn['RMSE'].values)))
print('bgd: MAE={} MSE={} RMSE={}'.format(mean_mae_bgd, mean_mse_bgd, mean_rmse_bgd))
print('sgd: MAE={} MSE={} RMSE={}'.format(mean_mae_sgd, mean_mse_sgd, mean_rmse_sgd))
