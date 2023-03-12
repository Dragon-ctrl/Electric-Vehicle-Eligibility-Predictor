import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('preprocessed_data.csv')

# Drop unnecessary columns
df = df.drop(['VIN (1-10)', 'DOL Vehicle ID', 'Vehicle Location', 'Electric Utility', '2020 Census Tract'], axis=1)

# Drop rows with missing values
df = df.dropna()

# Separate numerical and categorical columns
num_cols = ['Model Year', 'Electric Range', 'Base MSRP', 'Legislative District']
cat_cols = list(set(df.columns) - set(num_cols) - set(['Clean Alternative Fuel Vehicle (CAFV) Eligibility']))

# Standardize numerical features
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# One-hot encode categorical features
df = pd.get_dummies(df, columns=cat_cols)

# Split the dataset into features and target
X = df.drop('Clean Alternative Fuel Vehicle (CAFV) Eligibility', axis=1)
y = df['Clean Alternative Fuel Vehicle (CAFV) Eligibility']
