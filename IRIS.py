# Import necessary libraries
import pandas as pd
import pickle
import warnings

# Machine Learning libraries
from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbors classifier
from sklearn.preprocessing import LabelEncoder  # Encode target labels with value between 0 and n_classes-1
from sklearn.model_selection import train_test_split  # Split arrays or matrices into random train and test subsets
from sklearn.metrics import accuracy_score  # Accuracy classification score
from sklearn.linear_model import LogisticRegression  # Logistic Regression classifier
from sklearn.tree import DecisionTreeClassifier  # Decision Tree classifier
from sklearn.naive_bayes import GaussianNB  # Gaussian Naive Bayes classifier

# Load your dataset
df = pd.read_csv("D:/ABC/iris.csv", names=['sl', 'sw', 'pl', 'pw', 'class'])
# Reading a CSV file into a DataFrame, providing column names

# Apply label encoding
le = LabelEncoder()
df['class'] = le.fit_transform(df['class'])
# Encode 'class' column values into numerical labels

# Divide the data into input and output
X = df.drop(columns=['class'])  # Features (input)
Y = df['class']  # Labels (output/target)

# Split the data (70% for training and 30% testing) for training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
# Split the dataset into training and testing sets

# KNN model
K = KNeighborsClassifier(n_neighbors=5)  # Create a KNN classifier with 5 neighbors
K.fit(X_train, Y_train)  # Train the KNN model
Y_pred_knn = K.predict(X_test)  # Make predictions on the test set
acc_knn = accuracy_score(Y_test, Y_pred_knn)  # Calculate accuracy
acc_knn = round(acc_knn * 100, 2)  # Round the accuracy to 2 decimal places
print("Accuracy in KNN is", acc_knn, "%")  # Print the accuracy

# Logistic Regression model
warnings.filterwarnings('ignore')  # Ignore warnings for better readability
L = LogisticRegression()  # Create a Logistic Regression classifier
L.fit(X_train, Y_train)  # Train the Logistic Regression model
Y_pred_lg = L.predict(X_test)  # Make predictions on the test set
acc_lg = accuracy_score(Y_test, Y_pred_lg)  # Calculate accuracy
acc_lg = round(acc_lg * 100, 2)  # Round the accuracy to 2 decimal places
print("Accuracy in Logistic Regression is", acc_lg)  # Print the accuracy

# Decision Tree model
D = DecisionTreeClassifier()  # Create a Decision Tree classifier
D.fit(X_train, Y_train)  # Train the Decision Tree model
Y_pred_dt = D.predict(X_test)  # Make predictions on the test set
acc_dt = accuracy_score(Y_test, Y_pred_dt)  # Calculate accuracy
acc_dt = round(acc_dt * 100, 2)  # Round the accuracy to 2 decimal places
print("Accuracy in Decision Tree is", acc_dt)  # Print the accuracy

# Naive Bayes model
N = GaussianNB()  # Create a Gaussian Naive Bayes classifier
N.fit(X_train, Y_train)  # Train the Naive Bayes model
Y_pred_nb = N.predict(X_test)  # Make predictions on the test set
acc_nb = accuracy_score(Y_test, Y_pred_nb)  # Calculate accuracy
acc_nb = round(acc_nb * 100, 2)  # Round the accuracy to 2 decimal places
print("Accuracy in Naive Bayes is", acc_nb)  # Print the accuracy

# Save the Logistic Regression model to a file
with open('iris.pkl', 'wb') as f:
    pickle.dump(L, f)
# Save the trained Logistic Regression model to a binary file using pickle

