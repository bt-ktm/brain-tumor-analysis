import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# load data
df = pd.read_csv('bt_dataset_t3.csv')


# drop image column - the actual images are not included
df = df.drop('Image', axis=1)

#how many null values?
print(df.isnull().sum())

# drop null values
df = df.dropna()

# check column types
print(df.dtypes)


# check how many tumors & non-tumors there are & if they are all labeled as 0 or 1
print(df['Target'].value_counts())

# how many duplicates are present?
print(df.duplicated().sum())


# find outliers and show rows with outlier values
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = ((df < (Q1 - 3 * IQR)) | (df > (Q3 + 3 * IQR)))
outlier_rows = df[outliers.any(axis=1)]
print(outlier_rows)
# not removing any outliers because there are over 400 rows with outliers - we can address this later if needed

# separate features & label
x = df.drop('Target', axis=1) # x = features
y = df['Target'] # y = tumor classification label 

# scale features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)