
#  Importing library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
# reading data file
df = pd.read_csv("LR.csv")

# checking data file
print("")
print("\n Top 5 data  : ")
print(df.head())

print("\n Data Description : ")
print(df.describe())
print("\n Data INFO : ")
print(df.info())

# Exploratory Data Analysis
# Relationship between Time on App and Length of Membership
sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=df)
plt.show()

# Relationship between Time on Website and Yearly Amount Spent
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=df)
plt.show()

# Relationship between Time on App and Yearly Amount Spent
sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=df)
plt.show()




# Distribution of various features
sns.pairplot(df)
plt.show()

# Relationship between Length of Membership and Yearly Amount Spent
sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=df)
plt.show()

# Fatures or attributes
X = df[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]

# Target Value Y = amount of money spend by customers
Y = df['Yearly Amount Spent']


# splitting data set into test and train
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=100)

# Training
# fitting linear regression model
lm = LinearRegression()
lm.fit(X_train,y_train)

# The coefficients
print('Coefficients: \n', lm.coef_)

# Predicting
predictions = lm.predict( X_test)

# plot of predicted and actual value
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()

# performance measure
print("-------------------------------------------------------------------")
print("Performance measure :")
print("-------------------------------------------------------------------")
print('  MAE:\t', metrics.mean_absolute_error(y_test, predictions))
print('  MSE:\t', metrics.mean_squared_error(y_test, predictions))
print('  RMSE:\t', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# plot of residuals
sns.distplot((y_test-predictions),bins=50);
plt.show()

# Conclusion
# We still want to figure out the answer to the original question, do we focus our efforst on mobile app or website development? Or maybe that doesn't even really matter, and Membership Time is what is really important.  Let's see if we can interpret the coefficients at all to get an idea.

print("")
print("-------------------------------------------------------------------")
print("Coeffecient of features : ")
print("-------------------------------------------------------------------")

coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
print(coeffecients)
print("-------------------------------------------------------------------")
