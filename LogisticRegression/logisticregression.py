import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Create function to generate data
def generate_data(num_points):
    data = []

    for i in range(num_points):
        age = np.random.randint(18, 77)
        income = np.random.randint(10000, 100000)
        purchased = np.random.randint(0, 2)

        data.append({'age': age, 'income': income, 'purchased': purchased})

    dataFrame = pd.DataFrame(data)
    print(dataFrame)
    print("any  row in  age column  that has a NaN value : ", dataFrame.age.isna().any())
    if dataFrame.age.isna().any():
        dataFrame.age.fillna(dataFrame.age.mean())

    X = dataFrame[['age', 'income']]
    y = dataFrame['purchased']

    """Note  that the  splitting and scaling  the data  in this case happen in dataFrame levels . Split data into 
    training and testing dataFrames.
    Again  to note is the fact that we can only create a dataFrame from an array of python dictionaries ."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create logistic regression model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # Make predictions
    new_data = pd.DataFrame([{'age': 30, 'income': 40000}])
    new_data_scaled = scaler.transform(new_data)

    prediction = model.predict(new_data_scaled)
    predictionTestResults = model.predict(X_test_scaled)
    print(predictionTestResults)
    print("The predicted value is : {}".format(prediction))

    # Plot data
    plt.scatter(X_test['age'], X_test['income'], c=y_test, cmap='bwr', alpha=0.6)
    plt.xlabel('Age')
    plt.ylabel('Income')
    plt.title('Scatter Plot of Age vs. Income')
    plt.show()


# Call function
if __name__ == '__main__':
    generate_data(100)
