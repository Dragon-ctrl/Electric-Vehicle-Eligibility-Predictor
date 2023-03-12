import pandas as pd

# Load the dataset
df = pd.read_csv('./Electric_Vehicle_Population_Data.csv')

# Drop unnecessary columns
df.drop(['VIN (1-10)', 'DOL Vehicle ID', 'Vehicle Location', 'Electric Utility', '2020 Census Tract'], axis=1, inplace=True)

# Drop rows with missing values
df.dropna(inplace=True)

# Convert categorical variables to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['County', 'City', 'State', 'Make', 'Model', 'Electric Vehicle Type'])

# Save preprocessed data to CSV file
df.to_csv('preprocessed_data.csv', index=False)

# Split the dataset into features and target
X = df.drop('Clean Alternative Fuel Vehicle (CAFV) Eligibility', axis=1)
y = df['Clean Alternative Fuel Vehicle (CAFV) Eligibility']
