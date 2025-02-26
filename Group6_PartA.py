# pip install matplotlib
# pip install seaborn
# pip install pandas
# conda install pandas
# pip install statsmodels
# pip install pandas statsmodels
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import pearsonr

# Read CSV file using a raw string for the path
datas = pd.read_csv("Fill your file path")

#%% piecharts
def plot_counts_discrete(column_name, xlabel):
    if column_name in datas.columns:
        if column_name == 'Class':
            valid_classes = ['Eco', 'Eco Plus', 'Business', 'Unknown']
            datas_filtered = datas[datas['Class'].isin(valid_classes)]
        else:
            datas_filtered = datas

        counts = datas_filtered[column_name].value_counts()
        total_counts = counts.sum()
        percentages = (counts / total_counts * 100).round(1)
        print(percentages)
        print(counts)
        plt.figure(figsize=(10, 6))
        colors = ['red', 'green', 'blue', 'orange']  # Define the colors starting from green, red, and two others
        plt.pie(percentages.values, labels=percentages.index, autopct='%1.1f%%',
                startangle=140, textprops={'fontsize': 10}, colors=colors)  # Set fontsize for labels and colors
        plt.title(f'Percent of {xlabel}', y=1.05)  # Move title higher
        plt.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.
        plt.show()

# Columns and their respective x-axis labels for plotting
columns_to_plot_categorical = {
    'satisfaction': 'Satisfaction Level',
    'Class': 'Class Name',
    'Type of Travel': 'Types of Travelers',
    'Customer Type': 'Types of Customers'
}

# Plot counts for each discrete column
for column, xlabel in columns_to_plot_categorical.items():
    plot_counts_discrete(column, xlabel)
#%%
# Function to plot counts for continuous columns
def plot_counts_continuous(column_name):
    if column_name in datas.columns:
        plt.figure(figsize=(10, 6))
        if column_name in ['Cleanliness', 'Inflight wifi service', 'Departure/Arrival time convenient']:
            counts = datas[column_name].value_counts().sort_index()
            sns.histplot(datas[column_name], kde=False, binwidth=1, discrete=True, color='lightblue')
            plt.plot(counts.index, counts.values, marker='o', color='red', linestyle='-')
        else:
            sns.histplot(datas[column_name], kde=True)

        plt.xlabel(column_name)
        plt.ylabel('Count')
        plt.title(f'Count of {column_name}')

        # Ensure x-axis starts from 0
        plt.xlim(left=0)

        plt.show()

# Function to plot counts for continuous columns within specified borders
def plot_counts_continuous_limited(column_name, lower_limit, upper_limit):
    if column_name in datas.columns:
        datas_filtered = datas[(datas[column_name] >= lower_limit) & (datas[column_name] <= upper_limit)]
        plt.figure(figsize=(10, 6))
        sns.histplot(datas_filtered[column_name], kde=True)
        plt.xlabel(column_name)
        plt.ylabel('Count')
        plt.title(f'Count of {column_name} (Filtered between {lower_limit} and {upper_limit})')
        plt.xlim(left=lower_limit, right=upper_limit)
        plt.show()

# Plot counts for each continuous column with possible trend lines for discrete data
# Columns for plotting continuous data
columns_to_plot_continuous = {
    'Age', 'Flight Distance', 'Cleanliness', 'Inflight wifi service',
    'Departure/Arrival time convenient', 'Departure Delay in Minutes', 'Arrival Delay in Minutes'
}

# Plot counts for each continuous column
for column in columns_to_plot_continuous:
    plot_counts_continuous(column)

# Plot 'Age', 'Departure Delay in Minutes', and 'Arrival Delay in Minutes' with original and limited data
plot_counts_continuous('Age')
plot_counts_continuous_limited('Age', 0, 80)

plot_counts_continuous('Departure Delay in Minutes')
plot_counts_continuous_limited('Departure Delay in Minutes', 1, 40)

plot_counts_continuous('Arrival Delay in Minutes')
plot_counts_continuous_limited('Arrival Delay in Minutes', 1, 40)
#%%
# Remove rows with NaN values in 'Age' and 'Flight Distance' columns
data_cleaned = datas.dropna(subset=['Age', 'Flight Distance'])

# Create scatter plot of 'Flight Distance' vs 'Age' (original)
plt.figure(figsize=(10, 6))
plt.scatter(x=data_cleaned['Age'], y=data_cleaned['Flight Distance'], s=5, alpha=0.6)  # Adjust alpha for transparency
plt.xlabel('Age')
plt.ylabel('Flight Distance')
plt.title('Scatter Plot of Flight Distance by Age (Original)')
plt.xlim(left=0)  # Ensure x-axis starts from 0
plt.ylim(bottom=0)  # Ensure y-axis starts from 0
plt.show()

# Create scatter plot of 'Flight Distance' vs 'Age' (limited to 0-80 for Age)
plt.figure(figsize=(10, 6))
data_age_limited = data_cleaned[data_cleaned['Age'] <= 80]  # Filter data for age <= 80
plt.scatter(x=data_age_limited['Age'], y=data_age_limited['Flight Distance'], s=5, alpha=0.6)  # Adjust alpha for transparency
plt.xlabel('Age')
plt.ylabel('Flight Distance')
plt.title('Scatter Plot of Flight Distance by Age (Age 0-80)')
plt.xlim(left=0)  # Ensure x-axis starts from 0
plt.ylim(bottom=0)  # Ensure y-axis starts from 0
plt.show()

#%%
# Remove rows with NaN values in 'Cleanliness' and 'Gender'
data_cleaned = datas.dropna(subset=['Cleanliness', 'Gender'])

# Calculate normalized counts
normalized_counts = (
    data_cleaned
    .groupby('Gender')['Cleanliness']
    .value_counts(normalize=True)
    .rename('proportion')
    .reset_index()
)

# Plot normalized counts with a custom color palette
plt.figure(figsize=(10, 6))
sns.barplot(
    data=normalized_counts,
    x='Cleanliness',
    y='proportion',
    hue='Gender',
    palette=sns.color_palette("Spectral", n_colors=2)
)
plt.xlabel('Cleanliness')
plt.ylabel('Proportion')
plt.title('Normalized Count of Cleanliness by Gender')
plt.legend(title='Gender')
plt.show()

#%%
# Remove rows with NaN values in 'Inflight wifi service' and 'Ease of Online booking'
data_cleaned = datas.dropna(subset=['Inflight wifi service', 'Ease of Online booking'])
# Calculate normalized counts
normalized_counts = (
    data_cleaned
    .groupby('Ease of Online booking')['Inflight wifi service']
    .value_counts(normalize=True)
    .rename('proportion')
    .reset_index()
)
# Plot normalized counts with a custom color palette
plt.figure(figsize=(10, 6))
sns.barplot(
    data=normalized_counts,
    x='Inflight wifi service',
    y='proportion',
    hue='Ease of Online booking',
    palette=sns.color_palette("viridis", n_colors=6)  # Adjust the number of colors as needed
)
plt.xlabel('Inflight wifi service')
plt.ylabel('Proportion')
plt.title('Normalized Count of Inflight wifi service by Ease of Online booking')
plt.legend(title='Ease of Online booking')
plt.show()



#%%
data_cleaned = datas.dropna(subset=['On-board service', 'Inflight service'])
service_counts = data_cleaned.groupby(['On-board service', 'Inflight service']).size().unstack(fill_value=0)

# Plot the clustered bar chart
service_counts.plot(kind='bar', figsize=(10, 6))
plt.xlabel('On-board service')
plt.ylabel('Count')
plt.title('Distribution of On-board and Inflight Service Ratings')
plt.legend(title='Inflight service', loc='upper left')
plt.xticks(rotation=0)
plt.show()

#%%
plt.scatter(x=datas['Departure Delay in Minutes'], y=datas['Arrival Delay in Minutes'])
plt.xlabel('Departure Delay in Minutes')
plt.ylabel('Arrival Delay in Minutes')
plt.title('Relationship between Departure and Arrival Delays')
plt.xlim(left=0) # makes sure that the graphs will start from 0
plt.ylim(bottom=0)
plt.show()
#%%
# Filter data for the specified classes
valid_classes = ['neutral or dissatisfied', 'satisfied']
data_filtered = datas[datas['satisfaction'].isin(valid_classes)]

# Group by 'Class' and calculate mean ratings for each attribute
grouped_data = data_filtered.groupby('satisfaction')[['Checkin service', 'Food and drink', 'Seat comfort']].mean().reset_index()

# Reshape data for plotting
melted_data = pd.melt(grouped_data, id_vars=['satisfaction'], value_vars=['Checkin service', 'Food and drink', 'Seat comfort'],
                      var_name='Attribute', value_name='Mean Rating')

# Plot the clustered bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x='satisfaction', y='Mean Rating', hue='Attribute', data=melted_data, palette='Set2')
plt.xlabel('satisfaction')
plt.ylabel('Mean Rating')
plt.title('Mean Ratings of Checkin service, Food and drink, Seat comfort')
plt.legend(title='Attribute')
plt.xticks(rotation=0)
plt.ylim(0, 5)  # Set y-axis limits from 0 to 5
plt.show()
#%%
plt.figure(figsize=(10, 6))
# Filter the data
filtered_data = datas[(datas['Departure Delay in Minutes'] >= 1) & (datas['Departure Delay in Minutes'] <= 300)]
# Create the box plot
sns.boxplot(x='satisfaction', y='Departure Delay in Minutes', data=filtered_data, palette='viridis')
plt.xlabel('Satisfaction')
plt.ylabel('Departure Delay in Minutes')
plt.title('Departure Delay in Minutes vs Satisfaction (1-300 Minutes)')
plt.show()
#%%
# Create a copy of the datas DataFrame
data_copy = datas.copy()

# Ensure 'Plane colors' and 'satisfaction' are treated as strings in the copy
data_copy['Plane colors'] = data_copy['Plane colors'].astype(str)
data_copy['satisfaction'] = data_copy['satisfaction'].astype(str)

# Define the order of Plane colors
plane_color_order = ['1', '2', '3']  # Replace with your specific order of Plane colors

# Group by 'satisfaction' and 'Plane colors' and count the occurrences
counts = data_copy.groupby(['satisfaction', 'Plane colors']).size().reset_index(name='count')

# Plot the results with specified Plane color order
plt.figure(figsize=(10, 6))
sns.barplot(x='satisfaction', y='count', hue='Plane colors', data=counts, palette='viridis', order=['neutral or dissatisfied', 'satisfied'], hue_order=plane_color_order)

plt.title('Distribution of Satisfaction by Plane Color')
plt.xlabel('Satisfaction')
plt.ylabel('Count')
plt.legend(title='Plane color')

plt.show()

#%%
leg_room_counts = datas['Leg room service'].value_counts(dropna=False)

print(leg_room_counts)

missing_count = datas['Leg room service'].isna().sum()
not_missing_count = datas['Leg room service'].notna().sum()

data_legroom_missing = {
    'Status': ['Missing', 'Not Missing'],
    'Count': [missing_count, not_missing_count]
}
status_df = pd.DataFrame(data_legroom_missing)

status_df['Percentage'] = (status_df['Count'] / status_df['Count'].sum()) * 100

print(status_df)

plt.figure(figsize=(8, 8))
plt.pie(status_df['Count'], labels=status_df['Status'], autopct='%1.1f%%', startangle=140)
plt.title('Proportion of Missing vs. Not Missing Leg Room Service Ratings')
plt.show()
#%%
Arrival_Delay_counts = datas['Arrival Delay in Minutes'].value_counts(dropna=False)
print(Arrival_Delay_counts)

#%%
# Select only numeric columns for correlation matrix
numeric_columns = datas.select_dtypes(include=[np.number])
# Calculate the correlation matrix
correlation_matrix = numeric_columns.corr()
# Set up the matplotlib figure with a smaller size
sns.set(style='white')
fig, ax = plt.subplots(figsize=[12, 10])
# Create a mask to display only the lower triangle of the matrix
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
# Create a diverging color palette
cmap = sns.diverging_palette(150, 0, as_cmap=True)
# Plot the heatmap
sns.heatmap(correlation_matrix, cmap='seismic', linewidth=2, linecolor='white', vmax=1, vmin=-1, mask=mask, annot=True, fmt='.2f', ax=ax, cbar_kws={"shrink": .8})
# Adjust the font size for the labels
ax.tick_params(axis='both', which='major', labelsize=10)
# Rotate the x-axis labels for better readability
plt.xticks(rotation=45, ha='right')
# Add a title to the heatmap
ax.set_title('Correlation Heatmap', weight='bold', fontsize=15)
# Adjust the layout to make it more centered and readable
plt.tight_layout()
# Display the heatmap
plt.show()


#%%
# General function to clean any column based on valid values (discrete)
def clean_column_discrete(df, column_name, valid_values):
    df[column_name] = df[column_name].apply(lambda x: x if x in valid_values else np.nan)
    return df

# General function to clean any column based on valid range (continuous)
def clean_column_continuous(df, column_name, valid_range):
    min_val, max_val = valid_range
    df[column_name] = df[column_name].apply(lambda x: x if min_val <= x <= max_val else np.nan)
    return df

# Define valid values for discrete columns
columns_to_clean_discrete = {
    'Gender': ['Female', 'Male'],
    'Customer Type': ['Loyal Customer', 'disloyal Customer'],
    'Type of Travel': ['Personal Travel', 'Business travel'],
    'Class': ['Eco', 'Eco Plus', 'Business', 'Unknown'],
    'satisfaction': ['neutral or dissatisfied', 'satisfied'],
}

# Define valid ranges for continuous columns
columns_to_clean_continuous = {
    'Age': [0, 120],
    'Flight Distance': [0, 15843],
    'Plane colors': [1, 3],
    'Inflight wifi service': [1, 5],
    'Departure/Arrival time convenient': [0, 5],
    'Ease of Online booking': [0, 5],
    'Gate location': [0, 5],
    'Food and drink': [0, 5],
    'Seat comfort': [0, 5],
    'On-board service': [0, 5],
    'Leg room service': [0, 5],
    'Baggage handling': [0, 5],
    'Checkin service': [0, 5],
    'Inflight service': [0, 5],
    'Cleanliness': [0, 5],
    'Departure Delay in Minutes': [0, 120],
    'Arrival Delay in Minutes': [0, 120]
}
# Clean each column using the general functions
for column, valid_values in columns_to_clean_discrete.items():
    datas = clean_column_discrete(datas, column, valid_values)

for column, valid_range in columns_to_clean_continuous.items():
    datas = clean_column_continuous(datas, column, valid_range)
#%%
# Function to identify and drop columns with more than 25% missing values, excluding the 'satisfaction' column
def drop_columns_with_high_missing(df, threshold=0.25):
    total_rows = len(df)
    for column in df.columns:
        if column == 'satisfaction':
            continue
        missing_count = df[column].isnull().sum()
        missing_percentage = missing_count / total_rows
        if missing_percentage > threshold:
            print(f"Column '{column}' has {missing_percentage:.2%} missing values. Deleting...")
            df.drop(column, axis=1, inplace=True)
    return df

# Drop columns with more than 25% missing values directly on the 'datas' DataFrame
datas = drop_columns_with_high_missing(datas, threshold=0.25)

# Example usage
print("DataFrame shape after dropping columns:", datas.shape)

#%%
# Check for missing values in 'satisfaction' and remove rows if any are found
datas = datas.dropna(subset=['satisfaction'])
# Check if there are more than 4 variables missing in the line, if so remove it
datas = datas[datas.isnull().sum(axis=1) <= 4]
num_rows = datas.shape[0]
print(f"Number of rows in the DataFrame: {num_rows}")

#%%
# Fill missing data with mean value of column
# List of columns to fill NaN values with their mean
columns_to_fill = ['Age', 'Flight Distance', 'Departure Delay in Minutes']
# Loop through each column and fill NaN values with the mean of the column
for column in columns_to_fill:
    datas[column] = datas[column].fillna(datas[column].mean())
# Verify that NaN values have been filled
print(datas[columns_to_fill].isnull().sum())
#%%
columns_to_fill = ['Departure/Arrival time convenient', 'Ease of Online booking', 'Gate location', 'Food and drink',
                   'Seat comfort','On-board service','Baggage handling','Checkin service',
                   'Inflight service','Cleanliness']
# Loop through each column and fill NaN values with the median of the column
for column in columns_to_fill:
    datas[column] = datas[column].fillna(datas[column].median())
# Verify that NaN values have been filled
print(datas[columns_to_fill].isnull().sum())

#%%
# Replace NaN values in 'Inflight wifi service' with 0
datas['Inflight wifi service'] = datas['Inflight wifi service'].fillna(0)
# Replace NaN values in 'Class' with Unknown
datas['Class'] = datas['Class'].fillna('Unknown')
# Replace NaN values in 'Gender' with Unknown
datas['Gender'] = datas['Gender'].fillna('Unknown')
#%%
# Fill missing 'Arrival Delay in Minutes' with the values from 'Departure Delay in Minutes'
datas['Arrival Delay in Minutes'] = datas.apply(
    lambda row: row['Departure Delay in Minutes'] if pd.isna(row['Arrival Delay in Minutes']) else row['Arrival Delay in Minutes'],
    axis=1
)
#%%
# Completes the variables according to the ratio in which they exist in the data
print(datas[['Customer Type', 'Type of Travel' ]].isnull().sum())
def fill_missing_with_distribution(df, column_name):
    # Calculate the percentage distribution of the existing values
    value_counts = df[column_name].value_counts(normalize=True)
    print(f"Percentage of each value in {column_name}:")
    print(value_counts * 100)

    # making random selections from a specified list of values
    def fill_value(value):
        if pd.isna(value):
            return np.random.choice(value_counts.index, p=value_counts.values)
        else:
            return value

    # Apply the function to fill missing values
    df[column_name] = df[column_name].apply(fill_value)
    return df
# Example usage:
datas = fill_missing_with_distribution(datas, 'Customer Type')
datas = fill_missing_with_distribution(datas, 'Type of Travel')
datas = fill_missing_with_distribution(datas, 'Plane colors')

# Verify that missing values have been filled
print(datas[['Customer Type', 'Type of Travel', 'Plane colors']].isnull().sum())

#%%
# Define bins for age groups
age_bins = [0, 18, 30, 40, 50, 60, 120]  # Example bins: 0-18, 19-30, 31-40, 41-50, 51-60, 61+

# Define labels for each bin
age_labels = ['0-18', '19-30', '31-40', '41-50', '51-60', '61+']

# Discretize 'Age' into age groups
datas['Age_Group'] = pd.cut(datas['Age'], bins=age_bins, labels=age_labels, right=False)

# Plot the distribution of age groups
plt.figure(figsize=(8, 5))
sns.countplot(x='Age_Group', data=datas, palette='viridis')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.title('Distribution of Age Groups')
plt.xticks(rotation=45)
plt.show()
#%%
# Calculate the mean of service-related ratings
datas['Overall_Service_Rating'] = datas[['Inflight service', 'On-board service','Checkin service','Cleanliness','Food and drink']].mean(axis=1)

#%%
# Remove the 'Plane colors' column from the DataFrame
datas.drop(columns=['Plane colors'], inplace=True)