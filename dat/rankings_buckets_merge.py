import numpy as np
import pandas as pd

users = pd.read_csv('../dat/user_table.csv')
users.drop(['NaiveGrouping', 'WardGrouping'], inplace=True, axis=1)

rankings = pd.read_csv('../dat/rankings.csv')

print(rankings.shape)
print(users.shape)

print(np.amin(rankings['CustomerID'].sort_values().values == users['CustomerID'].sort_values().values))

new = rankings.merge(users, how='inner', on='CustomerID')

new.to_csv('feature_vectors.csv', index=False)
