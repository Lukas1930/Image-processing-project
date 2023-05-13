import pandas as pd

# Load the CSV data
data = pd.read_csv('metadata.csv')

# Filter data to only rows where diagnostic is 'NEV' or 'MEL'
#data = data[data['diagnostic'].isin(['NEV', 'MEL'])]

# Convert boolean columns to numeric
for col in ['itch', 'grew', 'hurt', 'changed', 'bleed', 'elevation']: 
    data[col] = data[col].map({'True': 1, 'False': 0})

# Group by diagnostic and calculate the mean of each other column
mean = data.groupby('diagnostic').mean()
print(mean)

median = data.groupby('diagnostic').median()
print(median)

std = data.groupby('diagnostic').std()
print(std)