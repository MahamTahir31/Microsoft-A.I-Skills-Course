# Module -9
# TRAIN AND EVALUATE REGRESSION MODEL
# ___________________________________
import pandas as pd
# load the training dataset
# !wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/ml-basics/daily-bike-share.csv
bike_data = pd.read_csv('daily-bike-share.csv')
print(bike_data.head())

# bike_data['day'] = pd.DatetimeIndex(bike_data['dteday']).day
# print(bike_data.head(32))

# numeric_features = ['temp', 'atemp', 'hum', 'windspeed']
# bike_data[numeric_features + ['rentals']].describe()
#________________________________________________________________________________________________________________________
import matplotlib.pyplot as plt

# Get the label column
label = bike_data['Rentals']


# Create a figure for 2 subplots (2 rows, 1 column)
fig, ax = plt.subplots(2, 1, figsize = (9,12))

# Plot the histogram   
ax[0].hist(label, bins=100)
ax[0].set_ylabel('Frequency')

# Add lines for the mean, median, and mode
ax[0].axvline(label.mean(), color='magenta', linestyle='dashed', linewidth=2)
ax[0].axvline(label.median(), color='cyan', linestyle='dashed', linewidth=2)

# Plot the boxplot   
ax[1].boxplot(label, vert=False)
ax[1].set_xlabel('Rentals')

# Add a title to the Figure
fig.suptitle('Rental Distribution')

# Show the figure
plt.show()

#_____________________________________________________________________________________________________________________________
# Plot a histogram for each numeric feature
# numeric_features = ['temp', 'atemp', 'hum', 'windspeed']
# for col in numeric_features:
#     fig = plt.figure(figsize=(9, 6))
#     ax = fig.gca()
#     feature = bike_data[col]
#     feature.hist(bins=100, ax = ax)
#     ax.axvline(feature.mean(), color='magenta', linestyle='dashed', linewidth=2)
#     ax.axvline(feature.median(), color='cyan', linestyle='dashed', linewidth=2)
#     ax.set_title(col)
# plt.show()

#_______________________________________________________________________________________________________________________
import numpy as np

# plot a bar plot for each categorical feature count
# categorical_features = ['season','mnth','holiday','weekday','workingday','weathersit', 'day']

# for col in categorical_features:
#     counts = bike_data[col].value_counts().sort_index()
#     fig = plt.figure(figsize=(9, 6))
#     ax = fig.gca()
#     counts.plot.bar(ax = ax, color='steelblue')
#     ax.set_title(col + ' counts')
#     ax.set_xlabel(col) 
#     ax.set_ylabel("Frequency")
# plt.show()
#_____________________________________________________________________________________________________________________
# for col in numeric_features:
#     fig = plt.figure(figsize=(9, 6))
#     ax = fig.gca()
#     feature = bike_data[col]
#     label = bike_data['rentals']
#     correlation = feature.corr(label)
#     plt.scatter(x=feature, y=label)
#     plt.xlabel(col)
#     plt.ylabel('Bike Rentals')
#     ax.set_title('rentals vs ' + col + '- correlation: ' + str(correlation))
# plt.show()
#_____________________________________________________________________________________________________________________
# plot a boxplot for the label by each categorical feature
# for col in categorical_features:
#     fig = plt.figure(figsize=(9, 6))
#     ax = fig.gca()
#     bike_data.boxplot(column = 'rentals', by = col, ax = ax)
#     ax.set_title('Label by ' + col)
#     ax.set_ylabel("Bike Rentals")
# plt.show()
#______________________________________________________________________________________________________________________\
# Separate features and labels
# X, y = bike_data[['season','mnth', 'holiday','weekday','workingday','weathersit','temp', 'atemp', 'hum', 'windspeed']].values, bike_data['rentals'].values
# print('Features:',X[:10], '\nLabels:', y[:10], sep='\n')


