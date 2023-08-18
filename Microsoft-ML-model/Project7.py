# MODULE - 8 
# EXPLORE AND ANALYZE DATA WITH PYTHON 

import pandas as pd

# !wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/ml-basics/grades.csv
df_dogs = pd.read_csv('doggy-boot-harness.csv',delimiter=',',header='infer')
print(df_dogs.head())

# Handling Missing Values
print(df_dogs.isnull()) # identify which individual values are null

print(df_dogs.isnull().sum())

print(df_dogs[df_dogs.isnull().any(axis=1)])

df_dogs.harness_size = df_dogs.harness_size.fillna(df_dogs.harness_size.mean())
print(df_dogs)

df_dogs = df_dogs.dropna(axis=0, how='any')
print(df_dogs) # dropping null values

# Get the mean harness_size using to column name as an index
mean_harn_size = df_dogs['harness_size'].mean()

# Get the mean boot_size using the column name as a property (just to make the point!)
mean_boot_size = df_dogs.boot_size.mean()

# Print the mean study hours and mean grade
print('Average weekly harness size: {:.2f}\nAverage boot_size: {:.2f}'.format(mean_harn_size , mean_boot_size))

boots  = pd.Series(df_dogs['boot_size'] >= 30)
df_dogs = pd.concat([df_dogs, boots.rename("harness_size")], axis=1)
print(df_dogs)

print(df_dogs.groupby(df_dogs.sex).age_years.count())


# Create a DataFrame with the data sorted by Grade (descending)
df_dogs = df_dogs.sort_values('age_years', ascending=False)

# Show the DataFrame
print(df_dogs)