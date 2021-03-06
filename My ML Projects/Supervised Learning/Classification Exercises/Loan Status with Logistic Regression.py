# ------------------------------------------------------------------
# Build the Logistic Regression Model
# Predict the loan approval status based on 
# Gender, Marital Status, Credit History, Income and Loan Amount
# ------------------------------------------------------------------

# Import Libraries
import pandas as pd

# Read the data and Create a copy
LoanData = pd.read_csv("ClassExercise.csv")
LoanPrep = LoanData.copy()


# Find out columns with missing values
LoanPrep.isnull().sum(axis=0)


# Replace Missing Values. Drop the rows.
LoanPrep = LoanPrep.dropna()

# Drop irrelevant columns based on common sense
LoanPrep = LoanPrep.drop(['gender'], axis=1)

# Create Dummy variables
LoanPrep.dtypes
LoanPrep = pd.get_dummies(LoanPrep, drop_first=True)


# Normalize the data (Income and Loan Amount) Using StandardScaler
from sklearn.preprocessing import StandardScaler
scalar_ = StandardScaler()

LoanPrep['income'] = scalar_.fit_transform(LoanPrep[['income']])
LoanPrep['loanamt'] = scalar_.fit_transform(LoanPrep[['loanamt']])


# Create the X (Independent) and Y (Dependent) dataframes
# -------------------------------------------------------
Y = LoanPrep[['status_Y']]
X = LoanPrep.drop(['status_Y'], axis=1)


# Split the X and Y dataset into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = \
train_test_split(X, Y, test_size = 0.3, random_state = 1234, stratify=Y)

# Build the Logistic Regression model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(X_train, Y_train)


# Predict the outcome using Test data
Y_predict = lr.predict(X_test)

# Build the conufsion matrix and get the accuracy/score
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_predict)

score = lr.score(X_test, Y_test)


