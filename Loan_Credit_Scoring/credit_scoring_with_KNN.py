import random
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

data = []

for i in range(10):
    payment_history = random.randint(0, 10)
    credit_utilization = round(random.uniform(0, 1), 2)
    credit_inquiries = random.randint(0, 5)
    income = random.randint(20000, 100000)
    creditScore = random.randint(0, 700)
    age = random.randint(20, 70)
    employment_status = random.choice(['employed', 'unemployed'])
    data.append([payment_history, credit_utilization, credit_inquiries, income, age, employment_status, creditScore])

if __name__ == '__main__':
    columnNames = ["PHistory", "CredUtilization", "C_Enquiries", "Income", "age",
                   "EmpStatus", "Score"]
    dataFrame = pd.DataFrame(data, columns=columnNames)
    print(dataFrame)

    labelEncoder = LabelEncoder()

    # Fit and transform the data using the LabelEncoder
    encoded_data = labelEncoder.fit_transform(dataFrame.EmpStatus)
    print(encoded_data)

    # Create KNN model
    model = KNeighborsRegressor(n_neighbors=3)

    dependentVariable = dataFrame.iloc[:, :-1]
    inDependentVariable = dataFrame.iloc[:, -1]

    emp_status_df = pd.DataFrame(encoded_data, columns=['E_Status'])

    df = pd.concat([dependentVariable, emp_status_df], axis=1)

    finalDataFrame = df.drop(['EmpStatus'], axis=1)
    print(finalDataFrame)

    # Fit model to data
    model.fit(finalDataFrame, inDependentVariable)
    model.feature_names = ["PHistory", "CredUtilization", "C_Enquiries", "Income", "age",
                           "EmpStatus"]

    """Note that the predict function will always accept a 2D array IF ONLY WE ARE INPUTTING THE DATA FOR A WHOLE 
    OBJECT  , [[6, 0.12, 1, 95110, 43, 1]], this  is down for us by the numpy  reshape() function.Inserting a row 
    calls  for a 2D array , reading rows is also done in 2D EG data[data["Cluster"] == 1]"""

    new_borrower = np.array([6, 0.12, 1, 95110, 43, 1]).reshape(1, -1)
    predicted_credit_score = model.predict(new_borrower)

    print("Predicted credit score:", predicted_credit_score[0])
