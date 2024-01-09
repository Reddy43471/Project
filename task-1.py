# import machine learning models, specifically from the scikit-learn library (assuming you meant "scikit-learn" instead of "neascret"), you can do so by using the appropriate modules.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

# Specify the file path
file_path = "D:\\NEXUS\\Project-1\\Iris - Iris.csv"  # Replace with the actual path to your CSV file

# Load the dataset from the CSV file
iris_df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(iris_df.head())

# Split the data into training and testing sets
X = iris_df.drop('Species', axis=1)
y = iris_df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')

# Histograms for each feature
iris_df.hist(bins=20, figsize=(12, 8))
plt.suptitle('Histograms of Iris Features', y=1.02)
plt.show()

# Box plots for each feature by species
num_features = len(iris_df.columns) - 1
num_cols = 2
num_rows = (num_features + num_cols - 1) // num_cols

plt.figure(figsize=(12, 8))
for i, col in enumerate(iris_df.columns[:-1]):
    plt.subplot(num_rows, num_cols, i + 1)
    sns.boxplot(x='Species', y=col, data=iris_df)
    plt.title(f'Boxplot of {col} by Species')
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

# Pairplot to visualize relationships between features
sns.pairplot(iris_df, hue='Species', markers=['o', 's', 'D'])
plt.show()
# Selecting only numerical features for correlation matrix
numerical_features = iris_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

# Violin plots for each feature by species
plt.figure(figsize=(12, 8))
for i, col in enumerate(iris_df.columns[1:5]):  # Exclude the 'Id' column
    plt.subplot(2, 2, i + 1)
    sns.violinplot(x='Species', y=col, data=iris_df, palette='Set3')
    plt.title(f'Violin plot of {col} by Species')
plt.tight_layout()
plt.show()

# Correlation matrix
correlation_matrix = numerical_features.corr()

# Plotting the correlation matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()





