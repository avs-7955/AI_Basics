# Create a raw data of minimum 40 records of height and weight in above mentioned
# format and use Min-Max Normalization to normalize the weights in the range from
# (-1.0 to 1, 0) as well as use Z-score to normalize the weights.

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

names = ["Height", "Weight"]

data = pd.read_csv("data.csv", names=names)

data.head()

array = data.values
# Seperating the array.
weights = array[:, 0:1]
heights = array[:, 1]

# Using Min-Max Normalization
scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
rescaled_weights = scaler.fit_transform(weights)

# Summarizing the transformed data with precision
np.set_printoptions(precision=3)
print("Min-Max Normalization")
# Printing the first 5 rows to check the output.
print(rescaled_weights[:6])

# # Using Z-score
scaler_02 = StandardScaler()
rescaled_weights_02 = scaler_02.fit_transform(weights)

# Summarizing the transformed data with precision
np.set_printoptions(precision=3)
print("Z-score Normalization")
# Printing the first 5 rows to check the output.
print(rescaled_weights_02[:6])


print("Successfully Executed")
